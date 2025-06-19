import pandas as pd
import numpy as np
from datasets import Dataset, ClassLabel
from sklearn.utils.class_weight import compute_class_weight
import torch
import random
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import psycopg2


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior in CUDA operations (may slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fetch_data_from_db(folder_ids: list = None):
    # Authenticate to Azure Key Vault
    credential = DefaultAzureCredential()
    keyvault_name = "vaultarchivai"
    kv_uri = f"https://{keyvault_name}.vault.azure.net"
    keys_client = SecretClient(vault_url=kv_uri, credential=credential)

    # Fetch secrets
    db_user = keys_client.get_secret("DB-USER").value
    db_password = keys_client.get_secret("DB-PASSWORD").value
    db_host = keys_client.get_secret("DB-HOST").value
    db_name = keys_client.get_secret("DB-NAME").value

    # Connect to the PostgreSQL database
    cnx = psycopg2.connect(user= db_user, password=db_password, host=db_host, port=5432, database=db_name)
    print("Connected successfully!")
    cur = cnx.cursor()
    query = """
    SELECT p."FileId", p."RawText", f."FolderId"
    FROM public."Page" p
    JOIN public."Files" f ON p."FileId" = f."Id"
    """
    if folder_ids:
        placeholders = ','.join(['%s'] * len(folder_ids))
        query += f' WHERE f."FolderId" IN ({placeholders})'
        cur.execute(query, tuple(folder_ids))
    else:
        cur.execute(query)
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=colnames)
    return df

def preprocess_data_db(df: pd.DataFrame, seed: int = 42):
    """
    Load and preprocess the dataset for training.
    
    Args:
        data_path (str): Path to the CSV data file
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (hf_dataset, tokenized_dataset, class_weights_tensor, num_labels, id2label, label2id)
    """
    set_seed(seed)
    
    # Load and preprocess the data
    data_df = df.rename(columns={"RawText": "text", "FolderId": "label"})
    data_df = data_df[["text", "label"]]
    data_df = data_df.dropna()
    data_df['label'] = data_df['label'].astype(str)
    # Create label mappings
    my_labels = list(data_df["label"].unique())
    id2label = {i: label for i, label in enumerate(my_labels)}
    label2id = {label: i for i, label in id2label.items()}
    num_labels = len(my_labels)
    
    # Build the label encoder
    class_labels = ClassLabel(names=list(set(data_df["label"])))
    
    # Map string labels to IDs
    data_df["label"] = data_df["label"].apply(lambda x: label2id[x])

    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(data_df)
    hf_dataset = hf_dataset.shuffle(seed=seed)
    
    # Calculate class weights for balanced training
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(data_df["label"]),
        y=data_df["label"]
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    print(f"Number of labels: {num_labels}")
    print(f"Null values: {data_df.isnull().sum().sum()}")
    print(f"Dataset shape: {data_df.shape}")
    
    return hf_dataset, class_weights_tensor, num_labels, id2label, label2id


def load_and_preprocess_data(data_path: str, seed: int = 42):
    """
    Load and preprocess the dataset for training.
    
    Args:
        data_path (str): Path to the CSV data file
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (hf_dataset, tokenized_dataset, class_weights_tensor, num_labels, id2label, label2id)
    """
    set_seed(seed)
    
    # Load and preprocess the data
    data_df = pd.read_csv(data_path)
    data_df = data_df.rename(columns={"raw_text": "text"})
    data_df["label"] = data_df["custom_id"].str.split("/").str[0]
    data_df = data_df[["text", "label"]]
    data_df = data_df.dropna()
    
    # Create label mappings
    my_labels = list(data_df["label"].unique())
    id2label = {i: label for i, label in enumerate(my_labels)}
    label2id = {label: i for i, label in id2label.items()}
    num_labels = len(my_labels)
    
    # Build the label encoder
    class_labels = ClassLabel(names=list(set(data_df["label"])))
    
    # Map string labels to IDs
    data_df["label"] = data_df["label"].apply(lambda x: class_labels.str2int(x))
    
    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(data_df)
    hf_dataset = hf_dataset.shuffle(seed=seed)
    
    # Calculate class weights for balanced training
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(data_df["label"]),
        y=data_df["label"]
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    print(f"Number of labels: {num_labels}")
    print(f"Null values: {data_df.isnull().sum().sum()}")
    print(f"Dataset shape: {data_df.shape}")
    
    return hf_dataset, class_weights_tensor, num_labels, id2label, label2id


def tokenize_dataset(hf_dataset, tokenizer, max_length: int = 1048, test_size: float = 0.2):
    """
    Tokenize the dataset and split into train/test.
    
    Args:
        hf_dataset: Hugging Face dataset
        tokenizer: Tokenizer to use
        max_length (int): Maximum sequence length
        test_size (float): Fraction for test split
        
    Returns:
        tuple: (train_dataset, eval_dataset)
    """
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        return tokenized_inputs
    
    print("Tokenizing dataset...")
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
    
    # Remove the original 'text' column and set format to PyTorch tensors
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")
    
    # Split into train/test
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=test_size)
    
    return tokenized_dataset['train'], tokenized_dataset['test']