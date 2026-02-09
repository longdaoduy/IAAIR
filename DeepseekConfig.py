import os
from dataclasses import dataclass

@dataclass
class DeepseekConfig:
    """Embedding model configuration."""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_provider: str = "huggingface"  # "huggingface", "openai", "sentence-transformers"
    embedding_dim: int = 768
    batch_size: int = 32
    max_sequence_length: int = 512
    temperature: int = 0.7
    device: str = "auto"  # "cpu", "cuda", "auto"

    @classmethod
    def from_env(cls) -> 'DeepseekConfig':
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", cls.model_name),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", cls.batch_size)),
            device=os.getenv("EMBEDDING_DEVICE", cls.device),
        )

