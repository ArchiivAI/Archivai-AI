from app.train_app import ModelSaver, JinaAIClassificationConfig, JinaAIForSequenceClassification
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
)
import threading
import os

class ModelConfig:
    _model = None
    _tokenizer = None
    _lock = threading.Lock()

    @classmethod
    def load_model(cls, model_name=None):
        """Load model from disk - only call when you want to load existing checkpoint"""
        save_model = ModelSaver()
        model_dir = save_model.load_model(name=model_name)

        # Register the model and config
        AutoConfig.register("jina_ai_classification", JinaAIClassificationConfig)
        AutoModel.register(JinaAIClassificationConfig, JinaAIForSequenceClassification)

        cls._tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        cls._model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)

        return cls._model, cls._tokenizer
    
    @classmethod
    def get_model(cls, model_name=None):
        """Get the model, loading it if necessary"""
        with cls._lock:
            if cls._model is None or cls._tokenizer is None:
                cls._model, cls._tokenizer = cls.load_model(model_name)
            return cls._model, cls._tokenizer