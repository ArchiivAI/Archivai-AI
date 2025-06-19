from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
import psycopg2
import os
class ModelSaver:
    def __init__(self):
        # Authenticate to Azure Key Vault
        self.credential = DefaultAzureCredential()
        self.keyvault_name = "vaultarchivai"
        self.kv_uri = f"https://{self.keyvault_name}.vault.azure.net"
        self.keys_client = SecretClient(vault_url=self.kv_uri, credential=self.credential)
        self.connection_string = self.keys_client.get_secret("alyaa-blob-connection-string").value
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_name = "azureml"
        self.container_client = self.blob_service_client.get_container_client(self.container_name)

        # Fetch secrets
        self.db_user = self.keys_client.get_secret("DB-USER").value
        self.db_password = self.keys_client.get_secret("DB-PASSWORD").value
        self.db_host = self.keys_client.get_secret("DB-HOST").value
        self.db_name = self.keys_client.get_secret("DB-NAME").value

        # Connect to the PostgreSQL database
        self.cnx = psycopg2.connect(user=self.db_user, password=self.db_password, host=self.db_host, port=5432, database=self.db_name)
        print("Connected successfully!")
        self.cur = self.cnx.cursor()

    def save_model(self, name, run_id, checkpoint_name, log_history, best_metric):
        azure_link = f"ExperimentRun/dcid.{run_id}/{checkpoint_name}/artifacts/{checkpoint_name}"
        for eval in log_history:
            if 'eval_loss' in eval and 'eval_accuracy' in eval:
                if eval['eval_loss'] == best_metric:
                    best_accuracy = eval['eval_accuracy']
                    best_f1 = eval['eval_f1']
                    best_recall = eval['eval_recall']
                    best_precision = eval['eval_precision']
        
        # Save model metadata to the database
        self.cur.execute(
            "INSERT INTO models (name, azure_link, accuracy, precision, recall, f1) VALUES (%s, %s, %s, %s, %s, %s)",
            (name, azure_link, best_accuracy, best_precision, best_recall, best_f1)
        )
        self.cnx.commit()
        return {
            azure_link: azure_link,
            "accuracy": best_accuracy,
            "precision": best_precision,
            "recall": best_recall,
            "f1": best_f1
        }
    
    def download_model(self, blob_start_with, local_dir=os.path.expanduser("~/downloaded/")):
        """
        Downloads all blobs from the Azure Blob Storage container that start with the specified prefix.
        """
        os.makedirs(local_dir, exist_ok=True)
        print(f"Downloading blobs under prefix: {blob_start_with}/")

        blobs_list = self.container_client.list_blobs(name_starts_with=blob_start_with)
        if not blobs_list:
            print(f"No blobs found with prefix: {blob_start_with}")
            return local_dir
        # print(f"Found {len(list(blobs_list))} blobs to download.")
        for blob in blobs_list:
            blob_path = blob.name
            filename = os.path.basename(blob_path)
            local_path = os.path.join(local_dir, filename)

            print(f"Downloading {blob_path} to {local_path}")
            with open(local_path, "wb") as f:
                download_stream = self.container_client.download_blob(blob_path)
                f.write(download_stream.readall())
        return local_dir
    
    def load_model(self, name=None):
        # get azure link from the database
        if name:
            self.cur.execute("SELECT azure_link FROM models WHERE name = %s", (name,))
        else:
            self.cur.execute("SELECT azure_link FROM models ORDER BY id DESC LIMIT 1")

        azure_link = self.cur.fetchone()
        azure_link = azure_link[0] if azure_link else None
        # download the model from Azure Blob Storage
        local_dir = self.download_model(azure_link)
        return local_dir
