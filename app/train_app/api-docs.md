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
  "status": "completed",
  "message": "Training completed successfully",
  "result": {
    "Status": "Training completed successfully",
    "Run ID": "abc123",
    "Checkpoint Name": "checkpoint-100",
    "Best Metric": 0.95
  },
  "timestamp": "2023-10-01T12:00:00"
}
```

**Error Responses:**
- 500: Internal server error
