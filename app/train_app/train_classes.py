import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    PretrainedConfig
)
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
import inspect

# ----------------------------------------------------------------------------------
# 1. Define a Custom Configuration Class
# This class will store all the settings for your model. It's the "blueprint"
# that allows Hugging Face to save and load your model architecture correctly.
# ----------------------------------------------------------------------------------
class JinaAIClassificationConfig(PretrainedConfig):
    """
    Configuration class for a JinaAIForSequenceClassification model.

    Inherits from `PretrainedConfig` and holds parameters specific to your
    custom model, as well as forwarding standard ones like num_labels.
    """
    model_type = "jina_ai_classification"  # A unique name for your custom architecture

    def __init__(
        self,
        base_model_name_or_path="jinaai/jina-embeddings-v2-base-en",
        lora_main_params_trainable=False,
        lora_task_name=None,
        classifier_dropout=0.1,
        **kwargs
    ):
        """
        Args:
            base_model_name_or_path (str): The name or path of the base Jina model.
            lora_main_params_trainable (bool): Flag to pass to the base model's config.
            lora_task_name (str, optional): The name of the LoRA task to activate.
            classifier_dropout (float): Dropout probability for the final classifier.
            **kwargs: Forwards standard arguments like `num_labels`, `id2label`, `label2id`
                      to the parent PretrainedConfig.
        """
        # Pass standard arguments (like num_labels) to the parent class.
        # This is the key to letting **kwargs handle them automatically.
        super().__init__(**kwargs)

        self.base_model_name_or_path = base_model_name_or_path
        self.lora_main_params_trainable = lora_main_params_trainable
        self.lora_task_name = lora_task_name
        self.classifier_dropout = classifier_dropout

# ----------------------------------------------------------------------------------
# 2. Define the Custom Model Class
# This class inherits from PreTrainedModel, which gives it all the Hugging Face
# superpowers like .from_pretrained() and .save_pretrained().
# ----------------------------------------------------------------------------------
class JinaAIForSequenceClassification(PreTrainedModel):
    """
    A custom Jina AI model for sequence classification with a mean pooling head.
    This model is compatible with the Hugging Face Trainer.
    """
    # FIX for RecursionError: The name of the attribute holding the base model
    # must NOT conflict with the internal `base_model` property of PreTrainedModel.
    # We rename our attribute to `jina_model` and set the prefix to match.
    base_model_prefix = "jina_model"
    config_class = JinaAIClassificationConfig  # Link to your custom config class

    def __init__(self, config: JinaAIClassificationConfig):
        super().__init__(config)
        # The model's config is now self.config, inherited from PreTrainedModel
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --- LoRA Task Activation (Robust Method) ---
        self.lora_task_id = None
        if config.lora_task_name:
            # WORKAROUND: In some contexts, the _adaptation_map attribute is not found on
            # self.base_model when instantiated inside a custom class. To robustly get
            # the map, we create a temporary standalone instance that is guaranteed
            # to be constructed correctly.
            print("Temporarily loading base model to fetch LoRA adaptation map...")
            try:
                temp_model = AutoModel.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
                if hasattr(temp_model, '_adaptation_map') and config.lora_task_name in temp_model._adaptation_map:
                    self.lora_task_id = temp_model._adaptation_map[config.lora_task_name]
                    print(f"Successfully mapped LoRA task '{config.lora_task_name}' to task_id: {self.lora_task_id}")
                else:
                    print(f"Warning: LoRA task name '{config.lora_task_name}' not found in the model's adaptation map.")
                    print("Available adaptation tasks:", getattr(temp_model, '_adaptation_map', {}).keys())
                del temp_model  # Clean up memory
            except Exception as e:
                print(f"Could not dynamically determine LoRA adaptation map. Error: {e}")
                print("Proceeding without LoRA task activation.")
        
        # --- Load Base Model with Correct Configuration ---
        # 1. Load the configuration of the BASE Jina model
        base_config = AutoConfig.from_pretrained(
            config.base_model_name_or_path,
            trust_remote_code=True
        )

        # 2. Modify it with settings from our custom config
        base_config.lora_main_params_trainable = config.lora_main_params_trainable

        # 3. Load the base model using the modified config
        # FIX: Rename attribute to `self.jina_model` to avoid name collision.
        self.jina_model = AutoModel.from_pretrained(
            config.base_model_name_or_path,
            config=base_config,
            trust_remote_code=True
        )
        
        # --- Classification Head ---.
        classifier_hidden_size = base_config.hidden_size
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(classifier_hidden_size, config.num_labels)
        # Initialize weights for the new classifier head
        self.post_init()

    def _mean_pooling(self, last_hidden_state, attention_mask):
        """Performs mean pooling on the last hidden state of the model."""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_embeddings = sum_embeddings / sum_mask
        # Normalize the pooled embeddings
        return F.normalize(pooled_embeddings, p=2, dim=1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs # Accept extra arguments
    ):
        """
        Forward pass for the classification model.
        """
        # --- Prepare inputs for the base model ---
        batch_size = input_ids.shape[0]
        current_device = input_ids.device
        
        model_input_args = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        sig = inspect.signature(self.jina_model.forward)
        if 'token_type_ids' in sig.parameters and token_type_ids is not None:
            model_input_args['token_type_ids'] = token_type_ids

        # Add adapter mask for LoRA if configured
        if self.lora_task_id is not None:
            adapter_mask_for_batch = torch.full((batch_size,), self.lora_task_id, dtype=torch.int32, device=current_device)
            model_input_args['adapter_mask'] = adapter_mask_for_batch
            for _, value in model_input_args.items():
                if isinstance(value, torch.Tensor):
                    model_input_args[_] = value.to(current_device)
        
        # --- Get Base Model Output ---
        model_outputs = self.jina_model(**model_input_args)

        # --- Extract Last Hidden State (handles different output formats) ---
        if hasattr(model_outputs, 'last_hidden_state'):
            last_hidden = model_outputs.last_hidden_state
        elif torch.is_tensor(model_outputs):
            last_hidden = model_outputs
        elif isinstance(model_outputs, dict) and 'last_hidden_state' in model_outputs:
            last_hidden = model_outputs['last_hidden_state']
        elif isinstance(model_outputs, dict) and 'embeddings' in model_outputs:
            last_hidden = model_outputs['embeddings']
            if len(last_hidden.shape) == 2:
                 raise ValueError(f"Base model output seems to be already pooled (shape: {last_hidden.shape}). The custom mean pooling requires sequence output [batch, seq_len, hidden_dim].")
        else:
            raise TypeError(f"Could not find a valid 'last_hidden_state' or 'embeddings' tensor in the base model output. Output type: {type(model_outputs)}")
            
        # --- Pooling and Classification ---
        pooled_output = self._mean_pooling(last_hidden, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # --- Calculate Loss ---
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # --- Return Structured Output ---
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(model_outputs, 'hidden_states', None),
            attentions=getattr(model_outputs, 'attentions', None),
        )
    def predict(self, texts, tokenizer, max_len=128):
        """
        Performs inference on a single string or a list of strings.

        Args:
            texts (str or list[str]): The input text(s) to classify.
            tokenizer (PreTrainedTokenizer): The tokenizer for the model.
            max_len (int): The maximum sequence length for the tokenizer.

        Returns:
            dict or list[dict]: A dictionary (for single input) or a list of
                                dictionaries (for multiple inputs) with the
                                predicted 'label' and 'score'.
        """
        self.to(self._device)
        self.eval()

        # Ensure the input is a list
        is_single_input = isinstance(texts, str)
        if is_single_input:
            texts = [texts]

        # Tokenize the input texts
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        # Move tensors to the same device as the model
        inputs = {key: tensor.to(self._device) for key, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self(**inputs)
            logits = outputs.logits

        # Convert logits to probabilities and get the top prediction
        probabilities = F.softmax(logits, dim=1)
        confidence_scores, predicted_ids = torch.max(probabilities, dim=1)

        # Map the predicted IDs to their string labels
        predicted_labels = [self.config.id2label[pred_id.item()] for pred_id in predicted_ids]

        # Format the results
        results = [
            {"label": label, "score": score.item()}
            for label, score in zip(predicted_labels, confidence_scores)
        ]

        # Return a single dictionary if the input was a single string
        return results[0] if is_single_input else results
