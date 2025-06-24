from app.ocr_functions import extract_text
from app.metadata_extractor import MetadataExtractor
from app.metadata_extractor import MetadataRequest
from app.rag_service import RAGService
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import base64
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from pydantic import BaseModel
import traceback
from typing import Optional, List
import cohere
from app.src.utils.inference import prediction
from app.src.utils.building_model import train_model
from fastapi.responses import StreamingResponse
from app.src.utils.model_manager import ModelManager
from app.inference_module import ModelConfig, predict

# FastAPI application instance
app = FastAPI(
    title="ArchivAI THE BEST AI!",
    description="AI API for ArchivAI",
    version="1.0.0")


# config Cors
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://ccdtr14p-3000.uks1.devtunnels.ms",
    "https://syntaxsquad-ai.azurewebsites.net"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)


# Authenticate to Azure Key Vault
credential = DefaultAzureCredential()
keyvault_name = "vaultarchivai"
kv_uri = f"https://{keyvault_name}.vault.azure.net"
keys_client = SecretClient(vault_url=kv_uri, credential=credential)

# get Cohere credentials from Azure Key Vault
cohere_api_key = keys_client.get_secret("CO-API-KEY").value
cohere_endpoint = keys_client.get_secret("AZURE-ML-COHERE-EMBED-ENDPOINT").value

# making embeddings client 
co_embed = cohere.Client(
    api_key=cohere_api_key,
    base_url=cohere_endpoint,)


# Authenticate to Azure OpenAI
api_base = keys_client.get_secret("archivai-openai-base").value
api_key= keys_client.get_secret("archivaigpt4-key").value
deployment_name = 'archivaigpt4'
api_version = '2024-08-01-preview'
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}")

# load the model
ModelConfig.load_model()
# Metadata Extractor Setup
metadata_extractor = MetadataExtractor(
    client=client,
    model=deployment_name,
    system_prompt="""You are a metadata extraction assistant.

Your task is to read the provided document content and extract ONLY the specific fields that will be requested.

Instructions:
- Extract ONLY the fields listed.
- Return your entire response STRICTLY as a JSON object.
- Do NOT include any explanations, comments, or any text before or after the JSON.
- If a field is missing, leave it with null value.
- Always return all requested fields even if you have to set some fields as null.

Format Example:
{
  "field1": "value1",
  "field2": "value2",
  ...
}

"""
)

# RAG Service Setup
rag_service = RAGService(keys_client=keys_client)

class classification(BaseModel):
    target_class: str
    accuracy: float

class TrainModelRequest(BaseModel):
    folder_ids: Optional[List[int]] = None

# Pydantic models for RAG endpoints
class StoreDataRequest(BaseModel):
    document_text: str
    file_id: int
    image_url: Optional[str] = None

class RetrieveRequest(BaseModel):
    question: str
    k: Optional[int] = 10

class ClearDataRequest(BaseModel):
    file_id: List[int] | int | None = None  # If None, clear all documents

def classify_file_bytes(file_bytes: bytes) -> str:
    """
    Classify an image to one of the specified categories.

    :param file_bytes: Image data in bytes.
    :return: Category of the image.
    """
    # Open the image from bytes
    img = Image.open(BytesIO(file_bytes))
    # Create a buffer to hold the image data
    buffer = BytesIO()
    # Save the image to the buffer in its original format
    img_format = img.format  # Get the image format (e.g., PNG, JPEG)
    img.save(buffer, format=img_format)
    # Encode the image data to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # Create the base64 string with the appropriate data URI scheme
    img_b64_str = f"data:image/{img_format.lower()};base64,{img_base64}"
    prompt = """
        classify this document to one of these categories:
['Advertisement',
 'Email',
 'Form',
 'Letter',
 'Memo',
 'News',
 'Note',
 'Report',
 'Resume',
 'Scientific']
 
 And give me the accuracy of your decision from 0 "not sure" to 100 "Sure".
 if the provided image does not belong to any of the categories, please provide the closest one and because your are not sure, make the accuracy low.
    """
    # Pass the image and the prompt to the model
    response = client.beta.chat.completions.parse(
        model=deployment_name,
        messages=[
            { "role": "system", "content": "You Are a document classification client." },
            { "role": "user", "content": [  
                { 
                    "type": "text", 
                    "text": prompt
                },
                { 
                    "type": "image_url",
                    "image_url":
                    {
                        "url": img_b64_str
                    }
                }
            ] }
        ],
        max_tokens=2000,
        response_format=classification)
    result = response.choices[0].message.parsed
    print(result)

    folder = result.target_class
    accuracy = result.accuracy
    section = "/SyntaxSquad/"
    return section + folder, accuracy

@app.get("/")
def hello_world():
    return {"message":"Hello World!ٌ load model first!"}

# ========== OCR ENDPOINTS ==========

@app.post("/extract-text/", tags=["OCR"])
async def extract_text_from_image(url: str = "None", file: Optional[UploadFile] = None, is_url: int = 0):
    """
    Endpoint to extract text from an uploaded image or PDF.

    :param url: URL of the file to be processed.
    :param file: Image or PDF file uploaded by the user.
    :param is_url: Flag to indicate if the file is provided as a URL.
    :return: JSON response with the extracted text.
    """
    print("I'm IN!")
    if is_url:
        if url == "None":
            raise HTTPException(status_code=400, detail="URL is required.")
        try:
            text = await extract_text(file=url, url=True)
            # return JSONResponse(content={"text": text.markdown_text, "raw_text": text.raw_text})
        except ValueError as ve:
            raise HTTPException(status_code=500, detail=str(ve))
        except Exception as e:
            traceback_str = ''.join(traceback.format_exception(e))
            print(traceback_str)
            raise HTTPException(status_code=500, detail=f"An error occurred while processing the URL: {str(e)}")

    else:
        if file is None:
            raise HTTPException(status_code=400, detail="File is required if URL is not provided.")
        
        try:
            # Validate the uploaded file's content type
            if file.content_type not in ["image/png", "image/jpeg", "image/jpg", "application/pdf"]:
                raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPEG, and PDF are supported.")

            # Read the file bytes
            file_bytes = await file.read()

            # Check the size of the file (limit to 5MB)
            if len(file_bytes) > 5 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="File size exceeds 5MB limit.")

            # Extract text from the file bytes
            text = await extract_text(file_bytes, url=False)

        except ValueError as ve:
            raise HTTPException(status_code=500, detail=str(ve))
        except Exception as e:
            # Log the exception details if necessary
            traceback_str = ''.join(traceback.format_exception(e))
            print(traceback_str)
            raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")
        
    # Convert the list of page objects to a list of dictionaries
    text_dicts = [page_obj.dict() for page_obj in text]
    return text_dicts

# ========== METADATA EXTRACTION ENDPOINTS ==========

@app.post("/extract_metadata", tags=["Metadata Extraction"])
async def extract_metadata_endpoint(request: MetadataRequest):
    try:
        result = metadata_extractor.extract_metadata(request.content, request.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== TRAINING ENDPOINTS ==========

@app.get("/load-model", tags=["Training"])
async def load_model_endpoint():
    """
    Endpoint to load the model from disk.
    This endpoint is used to load the model after training.
    Returns:
        JSON response with model loading status
    """
    try:
        ModelConfig.load_model()
        return JSONResponse(content={"status": "Model loaded successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while loading the model: {str(e)}")

@app.post("/predict-file", tags=["Training"])
async def predict_file_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to classify an uploaded file.
    """
    try:
        # Validate the uploaded file's content type
        if file.content_type not in ["image/png", "image/jpeg", "image/jpg", "application/pdf"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPEG, and PDF are supported.")

        # Read the file bytes
        file_bytes = await file.read()

        # Check the size of the file (limit to 5MB)
        if len(file_bytes) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 5MB limit.")

        # Classify the file bytes
        path, accuracy, text_dict = await predict(file_bytes)

        return JSONResponse(content={"path": path, "accuracy": accuracy, "text_dicts": text_dict})

    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while classifying the file: {str(e)}")

# ========== RAG ENDPOINTS ==========

@app.post("/rag/store", tags=["RAG"])
async def store_document_endpoint(request: StoreDataRequest):
    """
    Endpoint to store document text in the vector database for semantic search.
    
    Args:
        request: StoreDataRequest containing document_text, file_id, and optional image_url
        
    Returns:
        JSON response with storage status
    """
    try:
        result = rag_service.store_data(request.document_text, request.file_id)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        else:
            return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while storing the document: {str(e)}")

@app.post("/rag/retrieve", tags=["RAG"])
async def retrieve_documents_endpoint(request: RetrieveRequest):
    """
    Endpoint to retrieve relevant documents and get LLM response based on a question.
    
    Args:
        request: RetrieveRequest containing question and optional k parameter
        
    Returns:
        JSON response with retrieved file IDs and LLM response
    """
    try:
        file_ids, llm_response = rag_service.retrieve(request.question, request.k)
        return JSONResponse(content={
            "file_ids": file_ids,
            "response": llm_response,
            "total_retrieved": len(file_ids)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during retrieval: {str(e)}")

@app.post("/rag/clear", tags=["RAG"])
async def clear_documents_endpoint(request: ClearDataRequest):
    """
    Endpoint to clear documents from the vector database.
    
    Args:
        request: ClearDataRequest containing optional file_id
                - If file_id is provided, only documents with that ID will be deleted
                - If file_id is None, all documents will be cleared
        
    Returns:
        JSON response with deletion status and count
    """
    try:
        result = rag_service.clear_data(request.file_id)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        else:
            return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while clearing documents: {str(e)}")

@app.get("/rag/stats", tags=["RAG"])
async def get_rag_stats():
    """
    Endpoint to get statistics about the RAG vector database.
    
    Returns:
        JSON response with collection statistics
    """
    try:
        stats = rag_service.get_collection_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while getting stats: {str(e)}")

# ========== ARCHIVED ENDPOINTS ==========

@app.post("/classify-image/", tags=["ARCHIVED"])
async def classify_image_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to classify an uploaded image.

    :param file: Image file uploaded by the user.
    :return: JSON response with the classification result.
    """
    try:
        # Validate the uploaded file's content type
        if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid image type. Only PNG and JPEG are supported.")

        # Read the image bytes
        file_bytes = await file.read()

        # Check the size of the image (limit to 5MB)
        if len(file_bytes) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image size exceeds 5MB limit.")

        # Classify the image
        path, accuracy = classify_file_bytes(file_bytes)

        return JSONResponse(content={"path": path, "accuracy": accuracy})

    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        # traceback_str = ''.join(traceback.format_exception(e))
        # print(traceback_str)
        # Log the exception details if necessary
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image: {str(e)}")

@app.post("/classify-file/", tags=["ARCHIVED"])
async def classify_file_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to classify an uploaded File.
    """
    try:
        # Validate the uploaded file's content type
        if file.content_type not in ["image/png", "image/jpeg", "image/jpg", "application/pdf"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPEG, and PDF are supported.")
        
        # Read the file bytes
        file_bytes = await file.read()
        
        # Check the size of the file (limit to 5MB)
        if len(file_bytes) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 5MB limit.")
        
        # Check if model is initialized, if not, try to initialize
        model, encoder = ModelManager.get_model()
        if model is None or encoder is None:
            try:
                ModelManager.initialize()
            except Exception as init_error:
                raise HTTPException(
                    status_code=503, 
                    detail=f"Model not available. Please train a model first. Error: {str(init_error)}"
                )
        
        # Classify the file bytes
        path, accuracy, text_dict = await prediction(file_bytes, embedding_client=co_embed)

        return JSONResponse(content={"path": path, "accuracy": accuracy, "text_dicts": text_dict})

    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while classifying the file: {str(e)}")
    
@app.post("/Train-Model", tags=["ARCHIVED"])
async def train_model_endpoint(folder_ids: Optional[list[int]] = None):
    """
    Endpoint to train the model.

    """
    def massage_generator():
        
        try:
            # Check if folder_ids is None or a placeholder [0]
            if folder_ids is None or folder_ids == [0]:
                folder_ids_to_use = None
            else:
                folder_ids_to_use = folder_ids

            # Train the model using the Cohere client
            for message in train_model(co_embed, folder_ids_to_use):
                yield message
        except Exception as e:
            yield f"An error occurred during training: {str(e)}"    # Return a streaming response
    return StreamingResponse(massage_generator(), media_type="text/plain")

if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)