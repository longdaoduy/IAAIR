import os
from dataclasses import dataclass

@dataclass
class SciBERTConfig:
    """Embedding model configuration."""
    model_name: str = "allenai/scibert_scivocab_uncased"
    model_provider: str = "huggingface"  # "huggingface", "openai", "sentence-transformers"
    embedding_dim: int = 768
    batch_size: int = 32
    max_sequence_length: int = 512
    device: str = "auto"  # "cpu", "cuda", "auto"

    @classmethod
    def from_env(cls) -> 'SciBERTConfig':
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", cls.model_name),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", cls.batch_size)),
            device=os.getenv("EMBEDDING_DEVICE", cls.device),
        )

