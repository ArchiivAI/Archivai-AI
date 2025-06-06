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
  "message": "Hello World! test metadata extraction"
}
```

---

### 2. `POST /classify-image/`

**Description:**
Classifies an uploaded image into one of the following categories:
- Advertisement
- Email
- Form
- Letter
- Memo
- News
- Note
- Report
- Resume
- Scientific

Returns the predicted category and the model's confidence (accuracy).

**Request:**
- Content-Type: `multipart/form-data`
- Body: Image file (PNG, JPEG, JPG, max 5MB)

**Example using `curl`:**
```bash
curl -X POST "http://<host>:8000/classify-image/" \
  -H "accept: application/json" \
  -F "file=@example.jpg"
```

**Response Example:**
```json
{
  "path": "/SyntaxSquad/Report",
  "accuracy": 92.5
}
```

**Error Responses:**
- 400: Invalid image type or size exceeds 5MB
- 500: Internal server error

---

### 3. `POST /extract-text/`

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
    "raw_text": "Extracted raw text..."
  },
  ...
]
```

**Error Responses:**
- 400: Invalid file type, missing file or URL, or file size exceeds 5MB
- 500: Internal server error

---

### 4. `POST /extract_metadata`

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
### 5. `POST /classify-file/`

**Description:**  
This endpoint classifies an uploaded file (image or PDF) into a its folder ID. It is similar to `/classify-image/`, but supports both images and PDFs. In addition to returning the predicted folder ID (`path`) and model confidence (`accuracy`), it also returns extracted metadata or textual content in a dictionary (`text_dicts`).

The AI model analyzes the file content using embedded representations and document structure to predict the type and extract relevant fields, providing an enriched classification result. It uses Azure-powered AI under the hood for accurate and scalable predictions.

**Supported File Types:**
- PNG
- JPEG / JPG
- PDF

**File Size Limit:**
- Maximum file size is **5MB**

**Request:**
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Body Parameters:**
  - `file`: The file to be classified

**Example using `curl`:**
```bash
curl -X POST "http://<host>:8000/classify-file/" \
  -H "accept: application/json" \
  -F "file=@example.pdf"
  ```
**Successful Response Example:**
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
**Error Responses:**
- **400 Bad Request:**
  - Invalid file type (not PNG, JPEG, JPG, or PDF)
  - File size exceeds 5MB

- **500 Internal Server Error:**
  - Unexpected server error
  - File processing failure (e.g., model error or embedding issue)

----
### 6. `POST /Train-Model/`

**Description:**  
Triggers the training process for the document classification model. This endpoint reads labeled documents from the database, preprocesses them, optionally filters by specified folder IDs, encodes labels, trains a neural network classifier, and saves both the trained model and label encoder.

Progress is streamed live to the client during the training process.

**Request:**
- **Method:** `POST`
- **Content-Type:** `application/json`
- **Body Parameters (optional):**
  - `folder_ids`: (list of integers) A list of folder IDs to restrict training to specific labeled folders. If not provided, the model trains on all available labeled data.

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

**Error Streaming Example:**
```
Training started...


An error occurred during training: Database connection failed.
```

**Response Type:**  
- `text/event-stream` — The response is streamed progressively as plain text chunks.

**Error Responses:**
- **400 Bad Request:**
  - No data found for training
  - Invalid folder ID list
- **500 Internal Server Error:**
  - Unexpected failure during preprocessing, training, or saving

---

## RAG API Endpoints

This section describes the API endpoints for the Retrieval Augmented Generation (RAG) service, which allows storing, querying, and managing documents in a vector database for semantic search and question answering.

### 1. `POST /rag/store`

**Description:**
Stores document text and its associated `file_id` into the vector database. This makes the document content searchable for retrieval.

**Request Body:**
- Content-Type: `application/json`
```json
{
  "document_text": "The full text content of the document.",
  "file_id": 123,
  "image_url": "Optional URL to an image related to the document."
}
```

**Response Example (Success):**
```json
{
  "status": "success",
  "message": "Document with file_id 123 added to the vector store.",
  "file_id": 123
}
```

**Error Responses:**
- **500 Internal Server Error:**
  ```json
  {
    "detail": "An error occurred while storing the document: <error_message>"
  }
  ```
  (Or, if error originates from `rag_service` directly and not caught as HTTPException)
  ```json
  {
    "status": "error",
    "message": "Failed to store document: <error_message>",
    "file_id": 123
  }
  ```

---

### 2. `POST /rag/retrieve`

**Description:**
Retrieves relevant document `file_id`s based on a natural language question and provides an answer generated by an LLM using the retrieved context.

**Request Body:**
- Content-Type: `application/json`
```json
{
  "question": "What are the main findings of the report?",
  "k": 5
}
```
- `k` (optional, default: 10): Number of top relevant documents to retrieve.

**Response Example (Success):**
```json
{
  "file_ids": [123, 456, 789],
  "response": "The main findings of the report indicate a significant increase in market share...",
  "total_retrieved": 3
}
```

**Error Responses:**
- **500 Internal Server Error:**
  ```json
  {
    "detail": "An error occurred during retrieval: <error_message>"
  }
  ```
  (If `rag_service.retrieve` itself returns an error tuple, the response might look like):
  ```json
  {
    "file_ids": [],
    "response": "Error during retrieval: <error_message>",
    "total_retrieved": 0
  }
  ```

---

### 3. `POST /rag/clear`

**Description:**
Clears documents from the vector database. Can clear all documents or documents associated with specific `file_id`s.

**Request Body:**
- Content-Type: `application/json`
```json
{
  "file_id": [123, 456]
}
```
- `file_id` (optional): Can be a single integer or a list of integers. If provided, only documents with these `file_id`s will be deleted. If `null` or omitted, all documents in the collection will be cleared.

**Response Example (Success - Specific `file_id`s):**
```json
{
  "status": "success",
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
