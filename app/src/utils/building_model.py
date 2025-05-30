import cohere
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from torch.utils.data import DataLoader, TensorDataset
from app.src.utils.classifier import TextClassifier
from app.src.utils.trainer import Trainer
from app.src.utils.config import encoder_path
import numpy as np
import psycopg2
import pandas as pd

def train_model(chere_client: cohere.Client, folder_ids: list = None):
    
    # Connect to the PostgreSQL database
    cnx = psycopg2.connect(user="archivai", password="Saad@2356925", host="archivai-database.postgres.database.azure.com", port=5432, database="postgres")
    print("Connected successfully!")
    cur = cnx.cursor()
    # Construct the query
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

    if df.empty:
        print("No data found for the specified FolderIds.")
        return

    # Group by FileId to get the full content of each file
    files_content = []
    labels = []
    for file_id, group in df.groupby('FileId'):
        content = '\n'.join(group['RawText'].astype(str))
        folder_id = (group['FolderId'].iloc[0])
        files_content.append(content)
        labels.append(folder_id)

    # get the embeddings
    raw_text = files_content
    embeddings = chere_client.embed(input_type= 'classification', texts= raw_text).embeddings

    # encoding the target variable 
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)

    # Save the trained encoder
    joblib.dump(encoder, encoder_path)

    # converting to torch tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_encoded)

    # splitting the data to train and test
    embeddings_train, embeddings_test, classes_train, classes_test = train_test_split(
                embeddings_tensor, labels_tensor, test_size=0.20, random_state=0)
    
    # Create TensorDataset
    train_dataset = TensorDataset(embeddings_train, classes_train)
    test_dataset = TensorDataset(embeddings_test, classes_test)

    # Create DataLoaders
    batch_size = 32  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    # inistantiating a model
    model = TextClassifier(num_classes = len(encoder.classes_))

    # choosing device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # calculating counts
    class_counts = np.bincount(labels_encoded)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    weights = weights / weights.sum()

    # setting the criterion, optimizer and schedular
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # training and validating the model
    trainer_inist = Trainer(model, criterion, optimizer, scheduler, device)

    # Train the model
    for message in trainer_inist.train(train_loader, test_loader, num_epochs=30, patience= 5):
       yield message

            
            
    