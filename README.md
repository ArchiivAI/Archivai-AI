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

## Notes
- All endpoints return errors in JSON format with a `detail` field.
- For classification and extraction endpoints, only PNG, JPEG, JPG, and PDF (where applicable) are supported, with a 5MB size limit.
- The API uses Azure OpenAI and Key Vault for secure and scalable AI operations.

---

## Contact
For questions or support, contact the ArchivAI team.
