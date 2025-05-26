from pydantic import create_model, BaseModel, ValidationError
from typing import Optional, Dict, List
from openai import AzureOpenAI

class MetadataRequest(BaseModel):
    content: str
    features: Dict[str, str]  # example: {"title": "str", "year": "int"}

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
            "int": int
        }

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
                response_model=DynamicModel
            )
            return completion.choices[0].message.parsed
        except ValidationError as e:
            print("Validation error:", e)
            raise
        except Exception as e:
            print("General error:", e)
            raise
