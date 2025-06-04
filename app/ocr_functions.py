from PIL import Image
from io import BytesIO
import base64
from mimetypes import guess_type
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
import base64
from io import BytesIO
import fitz
import asyncio
from urllib.parse import urlparse
import time
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage,
    UserMessage,
    TextContentItem,
    ImageContentItem,
    ImageUrl,
    ImageDetailLevel,
)
import io

credential = DefaultAzureCredential()
keyvault_name = "vaultarchivai"
kv_uri = f"https://{keyvault_name}.vault.azure.net"
keys_client = SecretClient(vault_url=kv_uri, credential=credential)

api_base = keys_client.get_secret("archivai-openai-base").value
api_key= keys_client.get_secret("archivaigpt4-key").value
deployment_name = 'archivaigpt4'
api_version = '2024-08-01-preview'

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}"
)

azure_ai_endpoint = keys_client.get_secret("azure-ai-service-endpoint").value
key = keys_client.get_secret("azure-ai-service-key").value

class Page(BaseModel):
    markdown_text: str = Field(..., description="The text extracted from the image in markdown format.")
    raw_text: str = Field(..., description="The raw text extracted from the image without any formatting.")
    language: Literal["english", "arabic", "other"] = Field(..., description="The language in which the text is written in the image.")

system_prompt = """
                You are an OCR client using your Vision Capabilities to perform your response and provide a clean and structured text without any notes.
                The attributes you need to provide are:
                - markdown_text: The text extracted from the image in markdown format.
                - raw_text: The raw text extracted from the image without any formatting.
                - language: The language in which the text is written in the image. The options are "english" or "arabic" or "others".
                """
user_prompt = """extract the text from the image and provide a clean and structured text. 
                your output format must be in markdown format, with the text extracted from the image. 
                output will be  in 'markdown' text with markdown rules like titles,headings,etc, nothing else.
                
                NOTES:
                1) in the output, don't write ```markdown``` or ```md``` or any other code block.
                
                2) don't add any notes from you, just out the text extracted from the image. without any additions
                
                3) Warning! Don't Add Notice section to tell me that the text is not clear or any other notes.

                4) even if the text is not clear, try to extract as much as possible, and don't provide any extra notes.
                5) output the raw text extracted from the image as well. this text will be used for embedding purposes.
                6) Extract the text in the language it is written in the image. The options are "english" or "arabic".
                7) I will provide you multiple images, extract the text from all the images and provide the output in the same format.

                """

def download_pdf(blob_client):
    pdf_bytes = blob_client.download_blob().readall()
    return pdf_bytes

def generate_messages(system_prompt, user_prompt, image_urls):
    return [[
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": user_prompt
            },
            # Add one dict per image url
            (
                {
                    "type": "image_url",
                    "image_url": {"url": url}
                }
            )
        ],
    }]
    for url in image_urls]

def task(message):
    try:
        completion = client.beta.chat.completions.parse(
            model=deployment_name,
            messages=message,
            response_format=Page,
            )
        page = completion.choices[0].message.parsed
    except Exception as e:
        page = Page(markdown_text="", raw_text="")
    return page

def convert_pdf_to_images(pdf_path, url=False, dpi=300):
    # Open the PDF file
    if url:
        pdf_document = fitz.open(stream=download_pdf(pdf_path), filetype="pdf")
    else:
        # if pdf_path is a string file path
        if isinstance(pdf_path, str):
            pdf_document = fitz.open(pdf_path)
        else:
            pdf_document = fitz.open(stream=pdf_path, filetype="pdf")
    images = []

    # Calculate the zoom factor based on the desired DPI
    zoom = dpi / 72  # 72 is the default DPI for PDFs
    mat = fitz.Matrix(zoom, zoom)

    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images

def image_pil_to_data_url(pil_image, mime_type="image/png"):
    """
    Converts a PIL Image into a base64-encoded data URL string.
    """
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")  # or "JPEG"
    base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_data}"

def pdf_to_image_urls(pdf_file_path, url=False):
    """
    1. Converts each PDF page to a PIL Image.
    2. Converts each image to a data URL string.
    3. Returns a list of these data URLs.
    """
    # Convert each page in the PDF to a PIL image
    pil_images = convert_pdf_to_images(pdf_file_path, url, dpi=100)
    # Convert each PIL image to a Base64 data URL
    image_urls = [
        image_pil_to_data_url(pil_img, mime_type="image/png")
        for pil_img in pil_images
    ]
    return image_urls

async def main(messages):
    tasks = [asyncio.to_thread(task, message) for message in messages]
    result = await asyncio.gather(*tasks)
    return result

async def pdf_ocr(pdf_path, url=False):
    time_start = time.time()
    images = pdf_to_image_urls(pdf_path, url)
    messages = generate_messages(system_prompt, user_prompt, images)
    time_end = time.time()
    print(f"Generated {len(images)} messages in {time_end - time_start:.2f} seconds")
    
    time_start = time.time()
    final_result = await main(messages)
    time_end = time.time()
    print(f"Completed OCR in {time_end - time_start:.2f} seconds")
    
    return final_result 

def blob_to_base64(blob):
    image_bytes = blob.download_blob().readall()
    image = Image.open(BytesIO(image_bytes))
    imaeg_url = image_pil_to_data_url(image)
    return imaeg_url
    # buffered = BytesIO()
    # image.save(buffered, format="PNG")
    # img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # return img_str

async def image_ocr(blob,url=False):
    if url:
        image_url = blob_to_base64(blob)
    else:
        image_url = image_pil_to_data_url(Image.open(blob))
    messages = generate_messages(system_prompt, user_prompt, [image_url])
    final_result = await main(messages)
    return final_result

async def extract_text(file, url=False):
    if url:
        parsed_url = urlparse(file)
        path_parts = parsed_url.path.lstrip('/').split('/')
        container_name = path_parts[0]
        blob_name = '/'.join(path_parts[1:])
    
        blob_service_client = BlobServiceClient.from_connection_string(keys_client.get_secret("blob-connection-string").value)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        if file.endswith(".pdf"):
            result = await pdf_ocr(blob_client, url=url)
        elif file.endswith((".png", ".jpg", ".jpeg")):
            result = await image_ocr(blob_client, url=url)
        else:
            raise ValueError("Unsupported file type")

    else:
        if isinstance(file, bytes):
            # Determine the file type from the bytes
            file_type = None
            try:
                # Try to open as an image
                Image.open(io.BytesIO(file))
                file_type = "image"
            except IOError:
                # If it fails, try to open as a PDF
                try:
                    fitz.open(stream=file, filetype="pdf")
                    file_type = "pdf"
                except:
                    raise ValueError("Unsupported file type")

            if file_type == "pdf":
                result = await pdf_ocr(file, url=url)
            elif file_type == "image":
                result = await image_ocr(io.BytesIO(file), url=url)
        else:
            if file.endswith(".pdf"):
                result = await pdf_ocr(file)
            elif file.endswith((".png", ".jpg", ".jpeg")):
                result = await image_ocr(file)
    return result

def vlm_ocr(endpoint, model_deployment, system_prompt, user_prompt, url):
    vlm_client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
        headers={"azureml-model-deployment": model_deployment},
    )

    response = vlm_client.complete(
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=[
                TextContentItem(text=user_prompt),
                ImageContentItem(
                    image_url=ImageUrl.load(
                        image_file=url,
                        image_format='jpg',
                        detail=ImageDetailLevel.HIGH,
                    )
                )
            ])
        ],
        model=model_deployment
    )
    return response.choices[0].message.content

def gpt_ocr_layout(image_url):
    completion = client.beta.chat.completions.parse(
        model=deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url":
                        {
                            "url": image_url
                        }
                }
            ]},
        ],
        response_format=Page,
    )

    result = completion.choices[0].message.parsed
    return result
