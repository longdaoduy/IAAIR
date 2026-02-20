"""
Configuration for CLIP model client.

Defines settings and parameters for the CLIP model
used for generating visual embeddings.
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class CLIPConfig:
    """Configuration for CLIP client."""
    
    # Model settings
    model_name: str = "openai/clip-vit-large-patch14"  # Available: RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14
    device: Optional[str] = None  # Will auto-detect if not specified
    
    # Processing settings
    batch_size: int = 32
    max_image_size: int = 1024  # Max image dimension before resizing
    
    # Cache settings
    cache_embeddings: bool = True
    cache_dir: str = "./cache/clip_embeddings"
    
    @classmethod
    def from_env(cls) -> 'CLIPConfig':
        """Create configuration from environment variables."""
        return cls(
            model_name=os.getenv("CLIP_MODEL_NAME", "ViT-B/32"),
            batch_size=int(os.getenv("CLIP_BATCH_SIZE", "32")),
            max_image_size=int(os.getenv("CLIP_MAX_IMAGE_SIZE", "1024")),
            cache_embeddings=os.getenv("CLIP_CACHE_EMBEDDINGS", "true").lower() == "true",
            cache_dir=os.getenv("CLIP_CACHE_DIR", "./cache/clip_embeddings")
        )