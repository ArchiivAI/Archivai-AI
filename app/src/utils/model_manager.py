import threading
import torch
import joblib
import os
from app.src.utils.classifier import TextClassifier
from app.src.utils.config import encoder_path, checkpoint_path

class ModelManager:
    _model = None
    _encoder = None
    _lock = threading.Lock()

    @classmethod
    def load_model(cls):
        """Load model from disk - only call when you want to load existing checkpoint"""
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder not found at {encoder_path}")
            
        cls._encoder = joblib.load(encoder_path)
        
        # Create model with correct number of classes
        model = TextClassifier(num_classes=len(cls._encoder.classes_))
        
        # Try to load checkpoint if it exists and has matching dimensions
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                
                # Check dimensions before loading
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    # Check output layer dimensions (adjust the key name based on your model)
                    output_layer_key = 'layer_stack.8.weight'  # Adjust this key name
                    if output_layer_key in state_dict:
                        saved_classes = state_dict[output_layer_key].shape[0]
                        expected_classes = len(cls._encoder.classes_)
                        
                        if saved_classes == expected_classes:
                            model.load_state_dict(checkpoint['model_state_dict'])
                            print(f"Loaded checkpoint with {saved_classes} classes")
                        else:
                            print(f"Dimension mismatch: checkpoint has {saved_classes} classes, "
                                  f"encoder has {expected_classes} classes. Using fresh model.")
                    else:
                        print("Could not verify checkpoint dimensions. Using fresh model.")
                        
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Using fresh model.")
        
        model.eval()
        cls._model = model
        return cls._model, cls._encoder

    @classmethod
    def initialize(cls):
        """Initialize the model manager - only loads from disk if files exist"""
        with cls._lock:
            if not os.path.exists(encoder_path):
                print("No trained model found. Please train a model first.")
                cls._model = None
                cls._encoder = None
                return
            cls._model, cls._encoder = cls.load_model()

    @classmethod
    def get_model(cls):
        """Get current model and encoder"""
        with cls._lock:
            return cls._model, cls._encoder

    @classmethod
    def update_model(cls, new_model=None, new_encoder=None):
        """Update the model and/or encoder - does NOT load from disk"""
        with cls._lock:
            if new_encoder is not None:
                cls._encoder = new_encoder
                print("Updated encoder")

            if new_model is not None:
                try:
                    new_model.eval()
                    cls._model = new_model
                    print("Updated model successfully")
                except RuntimeError as e:
                    print("Error during model update:", e)
                    return False
            
            return True
    
    @classmethod
    def clear_model(cls):
        """Clear the current model and encoder"""
        with cls._lock:
            cls._model = None
            cls._encoder = None
            print("Cleared model and encoder")