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

    # ── Batch methods ────────────────────────────────────────────────────

    def generate_image_embeddings_batch(
        self, images: List[Union[str, Image.Image, np.ndarray]], batch_size: int = 16
    ) -> List[Optional[List[float]]]:
        """Generate CLIP image embeddings for a batch of images.

        Args:
            images: List of images (file paths, PIL Images, or numpy arrays).
            batch_size: Number of images to process in a single forward pass.

        Returns:
            List of embedding vectors (None for failed items).
        """
        if not images:
            return []
        if not self.model and not self.initialize():
            return [None] * len(images)

        # Convert all inputs to PIL
        pil_images: List[Optional[Image.Image]] = []
        for img in images:
            try:
                if isinstance(img, str):
                    pil_images.append(Image.open(img).convert("RGB"))
                elif isinstance(img, np.ndarray):
                    pil_images.append(Image.fromarray(img))
                else:
                    pil_images.append(img)
            except Exception as e:
                self.logger.warning(f"Failed to load image for batch: {e}")
                pil_images.append(None)

        results: List[Optional[List[float]]] = [None] * len(images)

        # Process in sub-batches
        valid_indices = [i for i, p in enumerate(pil_images) if p is not None]
        for start in range(0, len(valid_indices), batch_size):
            batch_idx = valid_indices[start : start + batch_size]
            batch_pils = [pil_images[i] for i in batch_idx]
            try:
                inputs = self.processor(images=batch_pils, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    features = self._ensure_tensor(outputs)
                    features = features / features.norm(p=2, dim=-1, keepdim=True)

                embeddings = features.cpu().numpy()
                for j, idx in enumerate(batch_idx):
                    results[idx] = embeddings[j].tolist()
            except Exception as e:
                self.logger.error(f"Batch image embedding error: {e}")
                # Fallback: process individually
                for idx in batch_idx:
                    results[idx] = self.generate_image_embedding(pil_images[idx])

        return results

    def generate_text_embeddings_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Optional[List[float]]]:
        """Generate CLIP text embeddings for a batch of texts.

        Args:
            texts: List of input texts.
            batch_size: Number of texts to process in a single forward pass.

        Returns:
            List of embedding vectors (None for failed items).
        """
        if not texts:
            return []
        if not self.model and not self.initialize():
            return [None] * len(texts)

        results: List[Optional[List[float]]] = [None] * len(texts)

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                inputs = self.processor(
                    text=batch, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model.get_text_features(**inputs)
                    features = self._ensure_tensor(outputs)
                    features = features / features.norm(p=2, dim=-1, keepdim=True)

                embeddings = features.cpu().numpy()
                for j, idx in enumerate(range(start, start + len(batch))):
                    results[idx] = embeddings[j].tolist()
            except Exception as e:
                self.logger.error(f"Batch text embedding error: {e}")
                for idx in range(start, start + len(batch)):
                    results[idx] = self.generate_text_embedding(texts[idx])

        return results