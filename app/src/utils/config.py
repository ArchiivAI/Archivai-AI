import os
from dotenv import load_dotenv
import sys

# load .env file
load_dotenv(override=True)

# variables
AZURE_ML_COHERE_EMBED_ENDPOINT= os.getenv('AZURE_ML_COHERE_EMBED_ENDPOINT')
CO_API_KEY= os.getenv('CO_API_KEY')
API_SECRET_KEY = os.getenv('API_SECRET_KEY')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_PATH = os.path.join(BASE_DIR, 'assets')

# Add the project root to Python path
sys.path.append(str(BASE_DIR))

# get the encoder 
encoder_path = os.path.join(ASSETS_PATH, "label_encoder.pkl")

# get the latest checkpoint
checkpoint_path = os.path.join(ASSETS_PATH, "best_model.pth")
