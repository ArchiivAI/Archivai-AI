import os
import argparse
import logging
import numpy as np
import mlflow
import evaluate
from dotenv import load_dotenv
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState
)
from transformers.integrations import MLflowCallback
from train_app.train_classes import (
    JinaAIClassificationConfig,
    JinaAIForSequenceClassification
)
from train_app.data_preprocessing import load_and_preprocess_data, tokenize_dataset, preprocess_data_db, fetch_data_from_db
from train_app.model_saving import ModelSaver
from train_app.config import Config
import requests
# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename='training.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class CustomMLflowCallback(MLflowCallback):
    """Custom MLflow callback to handle logging efficiently."""
    
    def __init__(self):
        super().__init__()
        self._last_logged_ckpt = None

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero:
            run = self._ml_flow.active_run()
            if run is not None:
                run_id = run.info.run_id
                    # Save run_id to a file
                with open("run_id.txt", "w") as f:
                    f.write(run_id)
                os.environ["MLFLOW_RUN_ID_CAPTURED"] = run_id
                print(f"✅ Run ID saved before ending: {run_id}")
            
            if self._auto_end_run and self._ml_flow.active_run():
                self._ml_flow.end_run()


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    # Load metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }



def train_model(folder_ids: list = None, output_dir: str = "output/jina_classification", run_name: str = "jina_classification_training"):
    """
    Main training function.
    
    Args:
        folder_ids (list): List of folder IDs to fetch data from
        output_dir (str): Output directory for model checkpoints
        run_name (str): MLflow run name
    """
    # Initialize config
    config = Config()
    
    print("=== Starting Jina AI Classification Training ===")
    
    # 1. Load the data
    print("1. Loading data...")
    df = fetch_data_from_db(folder_ids)
    if df.empty:
        print("No data found for the specified FolderIds.")
        return  

    # Preprocess the data
    print("Preprocessing data...")
    hf_dataset, class_weights_tensor, num_labels, id2label, label2id = preprocess_data_db(df)
    print(f"Number of labels: {num_labels}")
    
    # 2. Create model configuration
    print("2. Creating model configuration...")
    model_config = config.create_model_config(num_labels, id2label, label2id)
    
    # 3. Load tokenizer
    print("3. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, trust_remote_code=True)
    
    # 4. Create model
    print("4. Creating custom model...")
    # Register custom classes
    AutoConfig.register("jina_ai_classification", JinaAIClassificationConfig)
    AutoModel.register(JinaAIClassificationConfig, JinaAIForSequenceClassification)
    
    model = JinaAIForSequenceClassification(config=model_config)
    
    # Ensure classification head is trainable
    for name, param in model.classifier.named_parameters():
        param.requires_grad = True
    for name, param in model.dropout.named_parameters():
        param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 5. Tokenize dataset
    print("5. Tokenizing dataset...")
    train_dataset, eval_dataset = tokenize_dataset(hf_dataset, tokenizer, config.MAX_LENGTH)
    
    if run_name is None:
        run_name = "jina_classification_training"
    
    # 7. Setup training arguments
    print("7. Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=config.LOG_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.NUM_EPOCHS,
        fp16=False,
        load_best_model_at_end=True,
        weight_decay=0.01,
        warmup_ratio=0.1,
        report_to='none',
        logging_steps=10,
        logging_first_step=True,
        log_level='info',
        disable_tqdm=False,
        seed=42,
        run_name=run_name,
    )
    
    # 8. Initialize trainer
    print("8. Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # For testing with smaller dataset
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[CustomMLflowCallback()],
    )
    
    # 9. Train the model
    print("9. Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Error occurred during training: {e}")
        return

    # 10. Save the final model
    print("10. Saving final model...")
    model_save = ModelSaver()
    run_id = os.getenv("MLFLOW_RUN_ID_CAPTURED")
    checkpoint_name = trainer.state.best_model_checkpoint.split('/')[-1]
    log_history = trainer.state.log_history
    best_metric = trainer.state.best_metric
    model_save.save_model(
        name=run_name,
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        log_history=log_history,
        best_metric=best_metric,
    )
    print("Training completed successfully!")
    
    # load the model
    url = "https://archivai-ai.azurewebsites.net/load-model"
    status = requests.get(url)
    print(f"Model load status: {status.status_code}, Response: {status.text}")
    return {
        "Status": "Training completed successfully",
        "Run ID": run_id,
        "Checkpoint Name": checkpoint_name,
        "Best Metric": best_metric,
        "Model Load Status": status.status_code,
        "Model Load Response": status.text
    }

def main():
    # Test the training function
    result = train_model()
    print(result)
    return result


if __name__ == "__main__":
    main()


"""
az ml compute show --name H100-Cluster --resource-group archivai --workspace-name archivai-ai --query identity.principalId -o tsv
"""