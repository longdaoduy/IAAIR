import os
import warnings
# Suppress any warnings for google AI packages
warnings.filterwarnings("ignore")

import google.genai as genai
from google.genai import types
from typing import Optional
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class GeminiConfig:
    """Configuration for Google Gemini AI integration."""

    def __init__(self):
        # Try to load environment variables from .env file
        try:
            load_dotenv()
        except ImportError:
            # dotenv not available, continue with system env vars
            pass
            
        self.api_key = "AIzaSyD4oOjeEaDVHjE2PKg8ztRDYypxI7UQnX8"
        self.model_name = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
        self.temperature = float(os.getenv('GEMINI_TEMPERATURE', '0.1'))
        self.max_tokens = int(os.getenv('GEMINI_MAX_TOKENS', '1000'))

        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables - Gemini routing will be disabled")

    def initialize_client(self) -> Optional[object]:
        """Initialize Gemini client using google.genai."""
        try:
            if not self.api_key:
                logger.error("Cannot initialize Gemini: API key not provided")
                return None

            # Initialize the client with API key
            client = genai.Client(api_key=self.api_key)

            logger.info(f"Gemini client initialized successfully")
            return client

        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            return None

    def generate_text(self, client, prompt):
        # Use the models.generate method in the new SDK
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )
        return response.text