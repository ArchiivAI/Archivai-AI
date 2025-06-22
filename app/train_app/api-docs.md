# ArchivAI API Documentation

This API provides endpoints for document classification, text extraction, and metadata extraction using AI models and Azure services.

## Base URL

```
http://<host>:8000/
```

---

## Endpoints

### 1. `GET /`

**Description:**
Returns a simple hello world message to verify the API is running.

**Response Example:**
```json
{
  "message": "Welcome to the Jina AI Classification API!"
}
```

---

### 2. `POST /train`

**Description:**
Starts training a model using the provided folder IDs and configuration.

**Request:**
- Content-Type: `application/json`
- Body:
  ```json
  {
    "folder_ids": [1, 2, 3],
    "output_dir": "output/jina_classification",
    "run_name": "jina_classification_training"
  }
  ```
  **Note:** All inputs are optional and can be null. Default values will be used if not provided.

**Response Example:**
```json
{
  "status": "in_progress",
  "message": "Training started as a background task",
  "result": {
    "task_id": "task_20231001120000"
  },
  "timestamp": "2023-10-01T12:00:00"
}
```

**Error Responses:**
- 400: Training is already in progress
- 500: Internal server error

---

### 3. `GET /training-status`

**Description:**
Fetches the current status of the training process.

**Response Example:**
```json
{
  "status": "in_progress"
}
```

**Error Responses:**
- 500: Internal server error

---

### 4. `POST /edit-config`

**Description:**
Updates the model configuration parameters.

**Request:**
- Content-Type: `application/json`
- Body:
  ```json
  {
    "batch_size": 32,
    "max_length": 512,
    "learning_rate": 0.001,
    "num_epochs": 10,
    "classifier_dropout": 0.1
  }
  ```
  **Note:** All inputs are optional. Only provided parameters will be updated.

**Response Example:**
```json
{
  "status": "success",
  "message": "Configuration updated successfully",
  "updated_config": {
    "BATCH_SIZE": 32,
    "MAX_LENGTH": 512,
    "LEARNING_RATE": 0.001,
    "NUM_EPOCHS": 10,
    "CLASSIFIER_DROPOUT": 0.1
  }
}
```

**Error Responses:**
- 500: Internal server error
**Error Responses:**
- 500: Internal server error
