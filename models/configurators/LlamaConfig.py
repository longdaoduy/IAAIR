import os
import warnings
import logging
from typing import Optional, Union
from dotenv import load_dotenv
import requests

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
                
            # Create a simple client wrapper
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

            
            # Create a simple client wrapper
            class HuggingFaceClient:
                def __init__(self, model_name, temperature, max_tokens, token=None):
                    self.model_name = model_name
                    self.temperature = temperature
                    self.max_tokens = max_tokens
                    
                    logger.info(f"Loading Hugging Face model: {model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        use_auth_token=token,
                        trust_remote_code=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        use_auth_token=token,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True
                    )
                    
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                def generate_content(self, prompt):
                    try:
                        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                        if torch.cuda.is_available():
                            inputs = {k: v.cuda() for k, v in inputs.items()}
                            
                        with torch.no_grad():
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=self.max_tokens,
                                temperature=self.temperature,
                                do_sample=True,
                                pad_token_id=self.tokenizer.eos_token_id
                            )
                        
                        response_text = self.tokenizer.decode(
                            outputs[0][inputs['input_ids'].shape[1]:], 
                            skip_special_tokens=True
                        )
                        
                        return type('Response', (), {'text': response_text.strip()})()
                        
                    except Exception as e:
                        logger.error(f"HuggingFace generation error: {e}")
                        return None
            
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
            class MockClient:
                def __init__(self):
                    pass
                    
                def generate_content(self, prompt):
                    # Check if this is a routing decision prompt or a user question prompt
                    if "Query Type:" in prompt and "Routing Strategy:" in prompt and "academic paper search routing" in prompt:
                        # This is a routing decision prompt - provide routing analysis
                        prompt_lower = prompt.lower()
                        
                        if 'paper' in prompt_lower and ('id' in prompt_lower or 'w2' in prompt_lower):
                            response_text = """Query Type: STRUCTURAL
Routing Strategy: GRAPH_FIRST
Confidence: 0.9
Reasoning: Query contains specific paper ID, best handled by graph search for exact matches."""
                        elif 'author' in prompt_lower and ('wrote' in prompt_lower or 'by' in prompt_lower):
                            response_text = """Query Type: STRUCTURAL  
Routing Strategy: GRAPH_FIRST
Confidence: 0.8
Reasoning: Query asks about authorship, which requires graph traversal of author-paper relationships."""
                        elif any(word in prompt_lower for word in ['concept', 'topic', 'about', 'semantic', 'similarity']):
                            response_text = """Query Type: SEMANTIC
Routing Strategy: VECTOR_FIRST
Confidence: 0.8
Reasoning: Query involves conceptual similarity and topics, best handled by vector search."""
                        elif 'citation' in prompt_lower or 'reference' in prompt_lower:
                            response_text = """Query Type: HYBRID
Routing Strategy: PARALLEL
Confidence: 0.7
Reasoning: Citations involve both semantic content and structural relationships, needs both approaches."""
                        else:
                            response_text = """Query Type: HYBRID
Routing Strategy: PARALLEL
Confidence: 0.6
Reasoning: General query that may benefit from both vector and graph search approaches."""
                    else:
                        # This is a user question prompt - provide a helpful answer
                        prompt_lower = prompt.lower()
                        
                        if "search results:" in prompt_lower and "answer:" in prompt_lower:
                            # Extract the question from the prompt
                            if "who is the author" in prompt_lower:
                                response_text = """Based on the search results provided, I can help identify the authors of the paper. However, I need to examine the specific search results to provide accurate author information. If the search results contain the paper details, I would list the authors clearly. For example, if this were about the U-Net paper, the primary authors would typically include Olaf Ronneberger and colleagues from the University of Freiburg who developed this influential convolutional network architecture for biomedical image segmentation."""
                            elif "what is" in prompt_lower or "about" in prompt_lower:
                                response_text = """Based on the search results, I can provide information about the topic you're asking about. The search results should contain relevant papers and their abstracts that address your question. I'll synthesize the key findings and provide a comprehensive answer based on the available information."""
                            elif "how does" in prompt_lower or "how to" in prompt_lower:
                                response_text = """Based on the research papers found, I can explain the methodology or approach you're asking about. The search results should contain technical details and explanations that I can use to provide a clear, step-by-step answer to your question."""
                            else:
                                response_text = """Based on the search results provided, I can see relevant papers related to your question. Let me analyze the information and provide a comprehensive answer that addresses your specific query using the findings from these research papers."""
                        else:
                            response_text = "I'm a helpful research assistant ready to answer your questions based on academic search results."
                    
                    return type('Response', (), {'text': response_text})()
            
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