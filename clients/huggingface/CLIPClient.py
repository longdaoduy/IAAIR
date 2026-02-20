import logging
from typing import List, Optional, Union
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
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
            self.model = CLIPModel.from_pretrained(self.config.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.config.model_name)
            self.model.eval()  # Ensure model is in inference mode
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize CLIP model: {e}")
            return False

    def _ensure_tensor(self, output) -> torch.Tensor:
        """Helper to extract the raw tensor from HF output objects."""
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output

    def generate_image_embedding(self, image: Union[str, Image.Image, np.ndarray]) -> Optional[List[float]]:
        if not self.model and not self.initialize():
            return None

        try:
            if isinstance(image, str):
                pil_image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Extract features
                outputs = self.model.get_image_features(**inputs)
                # Safeguard against Wrapper Objects
                features = self._ensure_tensor(outputs)
                # Normalize (L2 Norm)
                features = features / features.norm(p=2, dim=-1, keepdim=True)

            return features.cpu().numpy().flatten().tolist()
        except Exception as e:
            self.logger.error(f"Image embedding error: {e}")
            return None

    def generate_text_embedding(self, text: str) -> Optional[List[float]]:
        if not self.model and not self.initialize():
            return None

        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                features = self._ensure_tensor(outputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)

            return features.cpu().numpy().flatten().tolist()
        except Exception as e:
            self.logger.error(f"Text embedding error: {e}")
            return None