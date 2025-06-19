from app.ocr_functions import extract_text
from app.train_app import ModelSaver, JinaAIClassificationConfig, JinaAIForSequenceClassification
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
)
from app.inference_module.model_config import ModelConfig
async def predict(file_bytes: bytes, model_name=None) -> tuple[int, float, str, str]:
    """
    Classify an image to one of the specified categories.

    :param file_bytes: Image data in bytes.
    :return: Tuple of (category_path, confidence, markdown_text, raw_text)
    """
    # Extract text from the file bytes
    result = await extract_text(file_bytes, url=False)

    # Combine all pages into one markdown and one raw text block
    full_raw = "\n".join(page.raw_text for page in result)

    # Convert the list of page objects to a list of dictionaries
    text_dicts = [page_obj.dict() for page_obj in result]

    # Get the model
    # save_model = ModelSaver()   
    # model_dir = save_model.load_model(name=model_name)
    
    # register the model
    model, tokenizer = ModelConfig.get_model()

    # Perform prediction using the model
    predicted = model.predict(full_raw, tokenizer=tokenizer)
    folder = predicted['label']
    accuracy = predicted['score']
    return int(folder), accuracy, text_dicts