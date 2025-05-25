from typing import List
from openai import AzureOpenAI
import json

# ================== Metadata Extractor Class ==================
class MetadataExtractor:
    def __init__(self, client: AzureOpenAI, model: str, system_prompt: str):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt

    def build_messages(self, content: str, features: List[str]) -> List[dict]:
        user_prompt = f"""
Here is the document content:

{content}

Please extract the following fields ONLY:
{', '.join(features)}

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

    def extract_metadata(self, content: str, features: List[str]) -> dict:
        messages = self.build_messages(content, features)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        result_raw = completion.choices[0].message.content
        try:
            result = json.loads(result_raw)
            return result
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            print(f"Raw output from model:\n{result_raw}")
            raise ValueError("The model output is not a valid JSON format.")
