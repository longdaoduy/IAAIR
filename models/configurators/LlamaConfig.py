import os
import logging
from typing import Optional, Union
from dotenv import load_dotenv

from clients.huggingface.OllamaClient import OllamaClient
from clients.huggingface.HuggingFaceClient import HuggingFaceClient
from clients.huggingface.MockClient import MockClient
import requests

logger = logging.getLogger(__name__)

class LlamaConfig:
    """Configuration for Llama AI integration using either Ollama or Hugging Face."""

    def __init__(self):
        # Try to load environment variables from .env file
        try:
            load_dotenv()
        except ImportError:
            # dotenv not available, continue with system env vars
            pass
            
        # Configuration options
        self.provider = os.getenv('LLAMA_PROVIDER', 'mock')  # 'ollama', 'huggingface', or 'mock'
        self.model_name = os.getenv('LLAMA_MODEL', 'llama2')
        self.temperature = float(os.getenv('LLAMA_TEMPERATURE', '0.1'))
        self.max_tokens = int(os.getenv('LLAMA_MAX_TOKENS', '1000'))
        
        # Ollama specific settings
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        # Hugging Face specific settings
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN', "")
        
        logger.info(f"LlamaConfig initialized with provider: {self.provider}, model: {self.model_name}")

    def initialize_client(self) -> Optional[object]:
        """Initialize Llama client based on configured provider."""
        try:
            if self.provider.lower() == 'ollama':
                return self._initialize_ollama_client()
            elif self.provider.lower() == 'huggingface':
                return self._initialize_huggingface_client()
            elif self.provider.lower() == 'mock':
                return self._initialize_mock_client()
            else:
                logger.error(f"Unsupported Llama provider: {self.provider}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to initialize Llama client: {e}")
            # Fallback to mock client for testing
            logger.info("Falling back to mock client for testing")
            return self._initialize_mock_client()

    def _initialize_ollama_client(self) -> Optional[object]:
        """Initialize Ollama client for local Llama models."""
        try:
            # Test connection to Ollama
            response = requests.get(f"{self.ollama_base_url}/api/version", timeout=5)
            if response.status_code != 200:
                logger.error(f"Cannot connect to Ollama server at {self.ollama_base_url}")
                logger.info("To use Ollama: 1) Install from https://ollama.ai 2) Run 'ollama serve' 3) Pull a model with 'ollama pull llama2'")
                return None

            client = OllamaClient(
                self.ollama_base_url, 
                self.model_name, 
                self.temperature, 
                self.max_tokens
            )
            
            logger.info(f"Ollama client initialized with model: {self.model_name}")
            return client
            
        except ImportError:
            logger.error("requests library not available for Ollama client")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            return None

    def _initialize_huggingface_client(self) -> Optional[object]:
        """Initialize Hugging Face client for Llama models."""
        try:

            client = HuggingFaceClient(
                self.model_name, 
                self.temperature, 
                self.max_tokens, 
                self.hf_token
            )
            
            logger.info(f"HuggingFace client initialized with model: {self.model_name}")
            return client
            
        except ImportError:
            logger.error("transformers or torch not available for HuggingFace client")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace client: {e}")
            return None

    def _initialize_mock_client(self) -> Optional[object]:
        """Initialize a mock client for testing purposes."""
        try:
            client = MockClient()
            logger.info("Mock client initialized for testing purposes")
            return client

        except Exception as e:
            logger.error(f"Failed to initialize mock client: {e}")
            return None

    def generate_text(self, client, prompt):
        """Generate text using the initialized client."""
        try:
            if client is None:
                logger.warning("No client available for text generation")
                return None
                
            response = client.generate_content(prompt)
            return response.text if response and hasattr(response, 'text') else None
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return None