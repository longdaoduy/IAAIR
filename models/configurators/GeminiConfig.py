import os
import google.generativeai as genai
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class GeminiConfig:
    """Configuration for Google Gemini AI integration."""
    
    def __init__(self):
        # Try to load environment variables from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            # dotenv not available, continue with system env vars
            pass
            
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model_name = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        self.temperature = float(os.getenv('GEMINI_TEMPERATURE', '0.1'))
        self.max_tokens = int(os.getenv('GEMINI_MAX_TOKENS', '1000'))
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables - Gemini routing will be disabled")
            
    def initialize_client(self) -> Optional[genai.GenerativeModel]:
        """Initialize Gemini client."""
        try:
            if not self.api_key:
                logger.error("Cannot initialize Gemini: API key not provided")
                return None
                
            genai.configure(api_key=self.api_key)
            
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            
            logger.info(f"Gemini client initialized with model: {self.model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            return None
