import logging
from typing import List, Optional, Union
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Assuming your config still provides the model name (e.g., "openai/clip-vit-base-patch32")
from models.configurators.CLIPConfig import CLIPConfig

class CLIPClient:
    """Client for generating CLIP embeddings using Hugging Face Transformers."""
    
    def __init__(self, config: Optional[CLIPConfig] = None):
        self.config = config or CLIPConfig()
        self.model = None
        self.processor = None
        self.device = None
        self.logger = logging.getLogger(__name__)
        
    def initialize(self) -> bool:
        """Initialize the Hugging Face CLIP model and processor."""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {self.device}")
            
            # Hugging Face uses from_pretrained instead of clip.load
            # Note: Ensure config.model_name is a HF-compatible path like 'openai/clip-vit-base-patch32'
            self.model = CLIPModel.from_pretrained(self.config.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.config.model_name)
            
            self.logger.info(f"CLIP model {self.config.model_name} loaded successfully via HF")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize CLIP model: {e}")
            return False
    
    def generate_image_embedding(self, image: Union[str, Image.Image, np.ndarray]) -> Optional[List[float]]:
        if not self.model:
            if not self.initialize(): return None
                
        try:
            # Standardize input to PIL
            if isinstance(image, str): pil_image = Image.open(image)
            elif isinstance(image, np.ndarray): pil_image = Image.fromarray(image)
            else: pil_image = image
            
            # Hugging Face processing
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return None

    def generate_text_embedding(self, text: str) -> Optional[List[float]]:
        if not self.model:
            if not self.initialize(): return None
                
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                # Normalize
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
            return text_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return None