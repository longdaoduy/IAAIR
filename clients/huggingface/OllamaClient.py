import requests
import logging

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url, model_name, temperature, max_tokens):
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_content(self, prompt):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            return type('Response', (), {'text': result.get('response', '')})()
        else:
            logger.error(f"Ollama API error: {response.status_code}")
            return None