from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

# Import training function
from train_app.train_script import train_model

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
