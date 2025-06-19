# ArchivAI Training App

## Note on Azure Container App
I have made a separate app for training endpoints so that I can run it on Azure Container App using NCA 100 GPU. This container app is serverless; it takes about 4 minutes to cold start to scale up from 0 nodes to 1, and takes 300 seconds (5 minutes) to scale down from 1 node to 0.

## File Structure

### 1. `train_script.py`
Contains the main training logic, including data preprocessing, model creation, and training execution.

### 2. `train_classes.py`
Defines custom classes for model configuration and architecture, tailored for Jina AI classification tasks.

### 3. `setup.py`
Handles Azure ML setup, including environment creation, data upload, and job submission.

### 4. `model_saving.py`
Manages saving trained models to Azure Blob Storage and PostgreSQL database, and downloading models for inference.

### 5. `data_preprocessing.py`
Includes functions for loading, preprocessing, and tokenizing datasets, with support for fetching data from a PostgreSQL database.

### 6. `config.py`
Provides configuration settings for the training process, including model parameters, Azure credentials, and MLflow integration.

### 7. `main.py`
The FastAPI entry point for the training app, exposing endpoints for training and monitoring.

### 8. `Dockerfile`
Defines the container environment for deploying the training app on Azure Container App.

### 9. `requirements.txt`
Lists Python dependencies required for the training app.

### 10. `__init__.py`
Initializes the `train_app` module and imports key classes.

### 11. `api-docs.md`
Documents the API endpoints exposed by the training app.

## Purpose
This app is designed to handle the training of AI models separately from the default application, ensuring scalability and efficient resource utilization on Azure.
