import cohere
import os
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from torch.utils.data import DataLoader, TensorDataset
from app.src.utils.classifier import TextClassifier
from app.src.utils.trainer import Trainer
from typing import List
from app.src.utils.config import encoder_path
from typing import List

def train_model(raw_text: List[str], labels : List[str]):

    # making embeddings client 
    co_embed = cohere.Client(
    api_key=os.getenv("AZURE_ML_COHERE_EMBED_CREDENTIAL"),
    base_url=os.getenv("AZURE_ML_COHERE_EMBED_ENDPOINT"),
     )
    
    # get the embeddings
    raw_text = raw_text
    embeddings = co_embed.embed(input_type= 'classification', texts= raw_text).embeddings

    # encoding the target variable 
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)

    # Save the trained encoder
    joblib.dump(encoder, encoder_path)

    # converting to torch tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_encoded, torch.int)

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
    model = TextClassifier(num_classes = len(labels))

    # choosing device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # calculating counts
    class_counts = labels.value_counts().sort_index()
    weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float32)
    weights = weights / weights.sum()

    # setting the criterion, optimizer and schedular
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # training and validating the model
    trainer_inist = Trainer(model, criterion, optimizer, scheduler, device)
    trainer_inist.train(train_loader, test_loader, num_epochs=30, patience= 5)

            
            
    