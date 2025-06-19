import os
from typing import Dict, Any, Optional
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from transformers import AutoTokenizer, AutoModel, AutoConfig
from train_app.train_classes import JinaAIClassificationConfig, JinaAIForSequenceClassification
import os
from pathlib import Path
class Config:
    """Configuration class for the Jina AI Classification project."""
    
    # Model configuration
    BASE_MODEL_NAME = "jinaai/jina-embeddings-v3"
    MAX_LENGTH = 1048
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    CLASSIFIER_DROPOUT = 0
    
    # Azure configuration
    KEYVAULT_NAME = "vaultarchivai"
    CONTAINER_NAME = "azureml"
    
    # MLflow configuration
    MLFLOW_EXPERIMENT_NAME = "jina-finetuned-custom_test_run_id"
    MLFLOW_TRACKING_URI = "azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/7e872089-24a0-424c-af5b-72396eef54c8/resourceGroups/archivai-ai/providers/Microsoft.MachineLearningServices/workspaces/alyaa-archivai-ai"
    
    # Data paths
    DATA_PATH = "extracted_data.csv"
    OUTPUT_DIR = "./outputs/jina_lora_out"
    LOG_DIR = "./outputs/jina_lora_logs"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_config = None
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup environment variables for MLflow and other services."""
        os.environ["MLFLOW_EXPERIMENT_NAME"] = self.MLFLOW_EXPERIMENT_NAME
        os.environ["MLFLOW_TRACKING_URI"] = self.MLFLOW_TRACKING_URI
        os.environ["MLFLOW_MAX_LOG_PARAMS"] = "180"
        os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "1"
        os.environ["MLFLOW_NESTED_RUN"] = "1"
        os.environ.pop("MLFLOW_RUN_ID", None)
    
    def load_model_from_azure_blob(self, run_id: str, checkpoint_name: str, local_dir: str = "./downloaded"):
        """
        Download and load the trained model from Azure Blob Storage.
        
        Args:
            run_id (str): MLflow run ID
            checkpoint_name (str): Name of the checkpoint (e.g., 'checkpoint-100')
            local_dir (str): Local directory to download the model
        """
        # Setup Azure credentials and blob client
        credential = DefaultAzureCredential()
        kv_uri = f"https://{self.KEYVAULT_NAME}.vault.azure.net"
        keys_client = SecretClient(vault_url=kv_uri, credential=credential)
        connection_string = keys_client.get_secret("alyaa-blob-connection-string").value
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(self.CONTAINER_NAME)
        
        # Define blob folder prefix
        blob_folder_prefix = f"ExperimentRun/dcid.{run_id}/{checkpoint_name}/artifacts/{checkpoint_name}"
        
        # Create local directory
        os.makedirs(local_dir, exist_ok=True)
        
        # Download all blobs with the given prefix
        print(f"Downloading model from Azure Blob Storage...")
        print(f"Blob prefix: {blob_folder_prefix}/")
        
        blobs_list = container_client.list_blobs(name_starts_with=blob_folder_prefix)
        
        for blob in blobs_list:
            blob_path = blob.name
            filename = os.path.basename(blob_path)
            local_path = os.path.join(local_dir, filename)
            
            print(f"Downloading {blob_path} to {local_path}")
            with open(local_path, "wb") as f:
                download_stream = container_client.download_blob(blob_path)
                f.write(download_stream.readall())
        
        print("Model download completed.")
        
        # Load the model and tokenizer
        self.load_model_from_local(local_dir)
    
    def load_model_from_local(self, model_path: str):
        """
        Load the model and tokenizer from local directory.
        
        Args:
            model_path (str): Path to the saved model directory
        """
        # Register custom model classes
        AutoConfig.register("jina_ai_classification", JinaAIClassificationConfig)
        AutoModel.register(JinaAIClassificationConfig, JinaAIForSequenceClassification)
        
        print(f"Loading model from {model_path}...")
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model_config = self.model.config
        
        print("Model and tokenizer loaded successfully!")
        print(f"Model type: {type(self.model)}")
        print(f"Number of labels: {self.model_config.num_labels}")
    
    def create_model_config(self, num_labels: int, id2label: Dict[int, str], label2id: Dict[str, int]):
        """
        Create a custom model configuration.
        
        Args:
            num_labels (int): Number of classification labels
            id2label (Dict[int, str]): Mapping from label IDs to label names
            label2id (Dict[str, int]): Mapping from label names to label IDs
            
        Returns:
            JinaAIClassificationConfig: Custom model configuration
        """
        self.model_config = JinaAIClassificationConfig(
            base_model_name_or_path=self.BASE_MODEL_NAME,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            lora_main_params_trainable=False,
            lora_task_name="classification",
            classifier_dropout=self.CLASSIFIER_DROPOUT
        )
        return self.model_config
    
    def get_model(self):
        """Get the loaded model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_from_azure_blob() or load_model_from_local() first.")
        return self.model
    
    def get_tokenizer(self):
        """Get the loaded tokenizer."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model_from_azure_blob() or load_model_from_local() first.")
        return self.tokenizer
    
    def get_model_config(self):
        """Get the model configuration."""
        if self.model_config is None:
            raise ValueError("Model config not created. Call create_model_config() or load model first.")
        return self.model_config

# Global config instance
config = Config()