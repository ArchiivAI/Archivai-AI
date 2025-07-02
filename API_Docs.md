# ArchivAI API Documentation

This API provides endpoints for document classification, text extraction, and metadata extraction using AI models and Azure services.

## Base URL

```
http://<host>:8000/
```

---

## Endpoints

### ========== OCR ENDPOINTS ==========

#### 1. `GET /`
**Description:**
Returns a simple hello world message to verify the API is running.

**Response Example:**
```json
{
  "message": "Hello World! test metadata extraction"
}
```

---

#### 2. `POST /extract-text/`
**Description:**
Extracts text from an uploaded image or PDF, or from a file at a given URL.

**Request:**
- Content-Type: `multipart/form-data` or `application/x-www-form-urlencoded`
- Body:
  - `file`: (optional) Image or PDF file (PNG, JPEG, JPG, PDF, max 5MB)
  - `url`: (optional) URL to the file
  - `is_url`: (int) 1 if using URL, 0 if uploading file

**Example 1: Uploading a file**
```bash
curl -X POST "http://<host>:8000/extract-text/" \
  -F "file=@example.pdf" \
  -F "is_url=0"
```

**Example 2: Using a URL**
```bash
curl -X POST "http://<host>:8000/extract-text/" \
  -F "url=https://example.com/document.jpg" \
  -F "is_url=1"
```

**Response Example:**
```json
[
  {
    "markdown_text": "Extracted text in markdown format...",
    "raw_text": "Extracted raw text...",
    "language": "english"
  },
  ...
]
```

**Notes:**
- The `language` field indicates the language of the extracted text. Possible values are:
  - `"english"`
  - `"arabic"`
  - `"other"`

**Error Responses:**
- 400: Invalid file type, missing file or URL, or file size exceeds 5MB
- 500: Internal server error

---

### ========== METADATA EXTRACTION ENDPOINTS ==========

#### 1. `POST /extract_metadata`
**Description:**
Extracts specified metadata fields from document content using AI.

**Request:**
- Content-Type: `application/json`
- Body:
  - `content`: (string) The document content to analyze
  - `features`: (dict) A dictionary mapping metadata field names to their types  
  - Allowed data types are:  
    1. `str` for string
    2. `int` for integer
    3. `float` for float
    4. `bool` for boolean
    5. `datetime.date` for date

**Example:**
```json
{
  "content": "This report was written by Salma in 2023-4-25.",
  "features": {
    "author": "str",
    "year": "int",
    "fulldate": "datetime.date"
  }
}
```

**Response Example:**
```json
{
  "author": "Salma",
  "year": 2023,
  "fulldate": "2023-4-25"
}
```

**Error Responses:**
- 500: Internal server error

---

### ========== TRAINING ENDPOINTS ==========

#### 1. `GET /load-model`
**Description:**
Loads the model from disk after training.

**Response Example:**
```json
{
  "status": "Model loaded successfully"
}
```

---

#### 2. `POST /predict-file`
**Description:**
Classifies an uploaded file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Image or PDF file (PNG, JPEG, JPG, PDF, max 5MB)

**Example using `curl`:**
```bash
curl -X POST "http://<host>:8000/predict-file/" \
  -F "file=@example.pdf"
```

**Response Example:**
```json
{
  "path": "/SyntaxSquad/Report",
  "accuracy": 92.5,
  "text_dicts": {
    "markdown_text": "Extracted text in markdown format...",
    "raw_text": "Extracted raw text..."
  }
}
```

**Error Responses:**
- 400: Invalid file type or size exceeds 5MB
- 500: Internal server error

---

### ========== RAG ENDPOINTS ==========

#### 1. `POST /rag/store`
**Description:**
Stores document text in the vector database for semantic search.

**Request Body:**
```json
{
  "document_text": "The full text content of the document.",
  "file_id": 123,
  "image_url": "Optional URL to an image related to the document."
}
```

**Example using `curl`:**
```bash
curl -X POST "http://<host>:8000/rag/store" \
  -H "Content-Type: application/json" \
  -d '{"document_text": "The full text content of the document.", "file_id": 123}'
```

**Response Example:**
```json
{
  "status": "success",
  "message": "Document with file_id 123 added to the vector store.",
  "file_id": 123
}
```

---

#### 2. `POST /rag/retrieve`
**Description:**
Retrieves relevant documents and generates an LLM response based on a question.

**Request Body:**
```json
{
  "question": "What are the main findings of the report?",
  "k": 5
}
```

**Example using `curl`:**
```bash
curl -X POST "http://<host>:8000/rag/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings of the report?", "k": 5}'
```

**Response Example:**
```json
{
  "file_ids": [123, 456, 789],
  "response": "The main findings of the report indicate a significant increase in market share...",
  "total_retrieved": 3
}
```

---

#### 3. `POST /rag/clear`
**Description:**
Clears documents from the vector database.

**Request Body:**
```json
{
  "file_id": [123, 456]
}
```

**Example using `curl`:**
```bash
curl -X POST "http://<host>:8000/rag/clear" \
  -H "Content-Type: application/json" \
  -d '{"file_id": [123, 456]}'
```

**Response Example (Success):**
```json
{
  "status": "success",
  "message": "Deleted 3 documents with file_id [123, 456]",
  "deleted_count": 3,
  "file_id": [123, 456]
}
```

---

#### 4. `GET /rag/stats`
**Description:**
Retrieves statistics about the RAG vector database collection.

**Example using `curl`:**
```bash
curl -X GET "http://<host>:8000/rag/stats"
```

**Response Example:**
```json
{
  "status": "success",
  "document_count": 150,
  "collection_name": "archivai_collection"
}
```

---

### ========== ARCHIVED ENDPOINTS ==========

<details>
<summary>Archived Endpoints</summary>

#### 1. `POST /classify-image/`
**Description:**
Classifies an uploaded image into predefined categories.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Image file (PNG, JPEG, JPG, max 5MB)

**Example using `curl`:**
```bash
curl -X POST "http://<host>:8000/classify-image/" \
  -F "file=@example.jpg"
```

**Response Example:**
```json
{
  "path": "/SyntaxSquad/Report",
  "accuracy": 92.5
}
```

---

#### 2. `POST /classify-file/`
**Description:**
Classifies an uploaded file (image or PDF) into its folder ID.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Image or PDF file (PNG, JPEG, JPG, PDF, max 5MB)

**Example using `curl`:**
```bash
curl -X POST "http://<host>:8000/classify-file/" \
  -F "file=@example.pdf"
```

**Response Example:**
```json
{
  "path": 501,
  "accuracy": 88.7,
  "text_dicts": {
    "markdown_text": "**Date:** 2023-04-25\n\n**From:** John Doe\n\n**Subject:** Meeting Confirmation",
    "raw_text": "Date: 2023-04-25\nFrom: John Doe\nSubject: Meeting Confirmation"
  }
}
```

---

#### 3. `POST /Train-Model`
**Description:**
Triggers the training process for the document classification model.

**Request:**
- Content-Type: `application/json`
- Body Parameters (optional):
  - `folder_ids`: (list of integers) A list of folder IDs to restrict training to specific labeled folders.

**Example using `curl`:**
```bash
curl -X POST "http://<host>:8000/Train-Model/" \
  -H "accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"folder_ids": [101, 102, 103]}'
```

**Streaming Response Example:**
```
Training started...


Training completed in 12.34 seconds.
```

</details>

---

## Notes
- All endpoints return errors in JSON format with a `detail` field.
- For classification and extraction endpoints, only PNG, JPEG, JPG, and PDF (where applicable) are supported, with a 5MB size limit.
- The API uses Azure OpenAI and Key Vault for secure and scalable AI operations.

---

## Contact
For questions or support, contact the ArchivAI team.
  "message": "Deleted 3 documents with file_id [123, 456]",
  "deleted_count": 3,
  "file_id": [123, 456]
}
```

**Response Example (Success - All documents cleared):**
```json
{
  "status": "success",
  "message": "All documents cleared from vector database.",
  "deleted_count": 150,
  "file_id": null
}
```

**Response Example (Warning - No documents found for `file_id`s):**
```json
{
  "status": "warning",
  "message": "No documents found with file_id [789]",
  "deleted_count": 0,
  "file_id": [789]
}
```

**Error Responses:**
- **500 Internal Server Error:**
  ```json
  {
    "detail": "An error occurred while clearing documents: <error_message>"
  }
  ```
  (Or, if error originates from `rag_service` directly)
  ```json
  {
    "status": "error",
    "message": "Failed to clear documents: <error_message>",
    "file_id": [123, 456]
  }
  ```

---

### 4. `GET /rag/stats`

**Description:**
Retrieves statistics about the RAG vector database collection, such as the total number of documents.

**Request Body:**
- None

**Response Example (Success):**
```json
{
  "status": "success",
  "document_count": 150,
  "collection_name": "archivai_collection"
}
```

**Error Responses:**
- **500 Internal Server Error:**
  ```json
  {
    "detail": "An error occurred while getting stats: <error_message>"
  }
  ```
  (Or, if error originates from `rag_service` directly)
  ```json
  {
    "status": "error",
    "message": "Failed to get collection stats: <error_message>"
  }
  ```

---

## Notes
- All endpoints return errors in JSON format with a `detail` field.
- For classification and extraction endpoints, only PNG, JPEG, JPG, and PDF (where applicable) are supported, with a 5MB size limit.
- The API uses Azure OpenAI and Key Vault for secure and scalable AI operations.

---

## Contact
For questions or support, contact the ArchivAI team.
