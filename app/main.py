from app.ocr_functions import extract_text
from app.metadata_extractor import MetadataExtractor
from app.metadata_extractor import MetadataRequest

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
from typing import Optional
from typing import List
from app.src.utils.inference import classify_file_bytes
from fastapi import FastAPI, HTTPException
import cohere

app = FastAPI(
    title="ArchivAI THE BEST AI!",
    description="AI API for ArchivAI",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://ccdtr14p-3000.uks1.devtunnels.ms",
    "https://syntaxsquad-ai.azurewebsites.net"
]

# config Cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    base_url=cohere_endpoint,
)

# Authenticate to Azure OpenAI
api_base = keys_client.get_secret("archivai-openai-base").value
api_key= keys_client.get_secret("archivaigpt4-key").value
deployment_name = 'archivaigpt4'
api_version = '2024-08-01-preview'
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}"
)
    
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

class classification(BaseModel):
    target_class: str
    accuracy: float


@app.get("/")
def hello_world():
    return {"message":"Hello World! test metadata extraction"}

@app.post("/classify-file/")
async def classify_file_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to classify an uploaded File.

    :param file: file uploaded by the user.
    :return: JSON response with the classification result.
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
        path, accuracy ,text_dict = classify_file_bytes(file_bytes, embedding_client = co_embed)

        return JSONResponse(content={"path": path, "accuracy": accuracy, "text_dicts": text_dict})

    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")

@app.post("/extract-text/")
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
            text = extract_text(file=url, url=True)
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
            text = extract_text(file_bytes, url=False)

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

@app.post("/extract_metadata")
async def extract_metadata_endpoint(request: MetadataRequest):
    try:
        result = metadata_extractor.extract_metadata(request.content, request.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)