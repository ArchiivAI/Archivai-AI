from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

# Import training function
from train_app.train_script import train_model, get_training_status
from train_app.config import config  # Import the global config instance

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Jina AI Classification API",
    description="API for training and inference with Jina AI classification models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for training requests
class TrainingRequest(BaseModel):
    folder_ids: Optional[List[int]] = Field(None, description="List of folder IDs to fetch data from")
    output_dir: Optional[str] = Field("output/jina_classification", description="Output directory for model checkpoints")
    run_name: Optional[str] = Field("jina_classification_training", description="MLflow run name")

class TrainingResponse(BaseModel):
    status: str
    message: str
    result: Optional[Dict[str, Any]] = None
    timestamp: datetime

class ConfigEditRequest(BaseModel):
    batch_size: Optional[int] = Field(None, description="Batch size for training")
    max_length: Optional[int] = Field(None, description="Maximum sequence length")
    learning_rate: Optional[float] = Field(None, description="Learning rate for training")
    num_epochs: Optional[int] = Field(None, description="Number of training epochs")
    classifier_dropout: Optional[float] = Field(None, description="Dropout rate for the classifier")


# Hello world endpoint
@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Welcome to the Jina AI Classification API!"}

@app.post("/train", response_model=TrainingResponse)
async def start_training(request: TrainingRequest):
    """
    Start training a model.
    
    - **folder_ids**: List of folder IDs to fetch training data from (optional)
    - **output_dir**: Directory to save model checkpoints (default: "output/jina_classification")
    - **run_name**: Name for the MLflow run (default: "jina_classification_training")
    """
    try:
        logger.info(f"Starting training with folder_ids: {request.folder_ids}, output_dir: {request.output_dir}, run_name: {request.run_name}")
        
        # Call the training function directly
        result = train_model(
            folder_ids=request.folder_ids,
            output_dir=request.output_dir,
            run_name=request.run_name
        )
        
        logger.info("Training completed successfully")
        
        return TrainingResponse(
            status="completed",
            message="Training completed successfully",
            result=result,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/training-status", tags=["Training"])
async def get_status():
    """
    Endpoint to get the current status of the training process.
    """
    status = get_training_status()
    return {"status": status}

@app.post("/edit-config", tags=["Config"])

async def edit_config(request: ConfigEditRequest):
    """
    Endpoint to edit the model configuration parameters.
    
    - **batch_size**: Batch size for training (optional)
    - **max_length**: Maximum sequence length (optional)
    - **learning_rate**: Learning rate for training (optional)
    - **num_epochs**: Number of training epochs (optional)
    - **classifier_dropout**: Dropout rate for the classifier (optional)
    """
    try:
        # Update configuration parameters if provided
        if request.batch_size is not None:
            config.BATCH_SIZE = request.batch_size
        if request.max_length is not None:
            config.MAX_LENGTH = request.max_length
        if request.learning_rate is not None:
            config.LEARNING_RATE = request.learning_rate
        if request.num_epochs is not None:
            config.NUM_EPOCHS = request.num_epochs
        if request.classifier_dropout is not None:
            config.CLASSIFIER_DROPOUT = request.classifier_dropout
        
        logger.info("Configuration updated successfully")
        
        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "updated_config": {
                "BATCH_SIZE": config.BATCH_SIZE,
                "MAX_LENGTH": config.MAX_LENGTH,
                "LEARNING_RATE": config.LEARNING_RATE,
                "NUM_EPOCHS": config.NUM_EPOCHS,
                "CLASSIFIER_DROPOUT": config.CLASSIFIER_DROPOUT
            }
        }
    except Exception as e:
        logger.error(f"Failed to update configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")
        logger.error(f"Failed to update configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

