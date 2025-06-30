import datetime
from pydantic import create_model, BaseModel, ValidationError
from typing import Optional, Dict, List
from openai import AzureOpenAI

class MetadataRequest(BaseModel):
    content: str
    features: Dict[str, str]

class MetadataExtractor:
    def __init__(self, client: AzureOpenAI, model: str, system_prompt: str):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt

    def build_messages(self, content: str, features: Dict[str, str]) -> List[dict]:
        user_prompt = f"""
                            Here is the document content:

                            {content}

                            Please extract the following fields ONLY:
                            {', '.join(features.keys())}

                            Instructions:
                            - Return the response STRICTLY as a valid JSON object.
                            - Do NOT add any explanations, comments, or any text before or after the JSON.
                            - If a field is missing, leave it with a null value.

                            Example format:
                            {{
                            "field1": "value1",
                            "field2": null,
                            ...
                            }}
                            """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def extract_metadata(self, content: str, features: Dict[str, str]) -> BaseModel:
        type_map = {
            "str": str,
            "string": str,
            "text": str,
            "integer": int,
            "int": int,
            "number": float,
            "float": float,
            "bool": bool,
            "datetime.date": datetime.date
        }

        # Handle custom types that are not in the type_map
        manual_features = {}
        for key, ftype in features.items():
            if ftype not in type_map:
                manual_features[key] = ftype

        # Remove manual features from the main features dictionary
        for key, _ in manual_features.items():
            del features[key]

        field_definitions = {
            key: (type_map.get(ftype, str), None)
            for key, ftype in features.items()
        }

        DynamicModel = create_model("DynamicMetadataModel", **field_definitions)

        messages = self.build_messages(content, features)

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=DynamicModel
            )

            # Parse the AI response into a dictionary, merging with manual features
            ai_features = completion.choices[0].message.parsed.dict()
            all_features = {**ai_features, **manual_features}
            return all_features
        
        except ValidationError as e:
            print("Validation error:", e)
            raise
        except Exception as e:
            print("General error:", e)
            raise
