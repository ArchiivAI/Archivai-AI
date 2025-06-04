import torch
import cohere
from app.ocr_functions import extract_text
from app.src.utils.model_manager import ModelManager




def prediction(file_bytes: bytes, embedding_client: cohere.Client) -> tuple[int, float, str, str]:
    """
    Classify an image to one of the specified categories.

    :param file_bytes: Image data in bytes.
    :return: Tuple of (category_path, confidence, markdown_text, raw_text)
    """
    # Extract text from the file bytes
    result = extract_text(file_bytes, url=False)

    # Combine all pages into one markdown and one raw text block
    full_raw = "\n".join(page.raw_text for page in result)

    # Convert the list of page objects to a list of dictionaries
    text_dicts = [page_obj.dict() for page_obj in result]

    # get the embeddings
    text = full_raw    
    embeddings_response = embedding_client.embed(input_type='classification', texts=[text])
    embeddings = embeddings_response.embeddings
 

    # creating a pytorch tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
   
    # set the model to evaluation mode
    model, encoder = ModelManager.get_model()
    
    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embeddings_tensor = embeddings_tensor.to(device)
    
    # Get predictions and confidence scores
    with torch.no_grad():
        outputs = model(embeddings_tensor)
        
        # Get class predictions
        _, predicted_classes = torch.max(outputs, 1)

        # Get names of the classes
        classes = encoder.inverse_transform(predicted_classes.cpu().numpy())

        # Get confidence scores (using softmax)
        confidences = torch.nn.functional.softmax(outputs, dim=1)
        confidence_values, _ = torch.max(confidences, 1)

        # Extract single value
        folder = classes[0] if len(classes) > 0 else "unknown"
        accuracy = confidence_values[0].item()

        return int(folder), accuracy, text_dicts