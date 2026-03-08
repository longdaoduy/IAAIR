import os
from dataclasses import dataclass, field
from typing import Dict, List


# Supported instruction-tuned LLMs (similar capability to Qwen2.5-1.5B-Instruct)
SUPPORTED_MODELS: Dict[str, Dict] = {
    # ── Small (≤ 1.5B) ── fast, low VRAM (~3–4 GB) ──
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "params": "1.5B", "vram_gb": 3.5, "speed": "fast",
        "notes": "Default. Good balance of speed and quality for routing/Cypher."
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "params": "0.5B", "vram_gb": 1.5, "speed": "very fast",
        "notes": "Ultra-light. Suitable for simple routing, weaker on generation."
    },
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": {
        "params": "1.7B", "vram_gb": 3.5, "speed": "fast",
        "notes": "Compact model with solid instruction following."
    },
    "microsoft/Phi-3.5-mini-instruct": {
        "params": "3.8B", "vram_gb": 8, "speed": "medium",
        "notes": "Strong reasoning, good for Cypher generation and synthesis."
    },
    "google/gemma-2-2b-it": {
        "params": "2B", "vram_gb": 5, "speed": "fast",
        "notes": "Google Gemma 2B instruction-tuned. Good general quality."
    },
    # ── Medium (3B–8B) ── better quality, more VRAM (~8–18 GB) ──
    "Qwen/Qwen2.5-3B-Instruct": {
        "params": "3B", "vram_gb": 7, "speed": "medium",
        "notes": "Higher quality Qwen. Better Cypher and synthesis."
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "params": "7B", "vram_gb": 16, "speed": "slow",
        "notes": "High quality. Needs ≥16 GB VRAM."
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "params": "8B", "vram_gb": 18, "speed": "slow",
        "notes": "Strong overall quality. Needs ≥18 GB VRAM."
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "params": "7B", "vram_gb": 16, "speed": "slow",
        "notes": "Excellent instruction following. Needs ≥16 GB VRAM."
    },
}


@dataclass
class DeepseekConfig:
    """LLM model configuration."""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_provider: str = "huggingface"  # "huggingface", "openai", "sentence-transformers"
    embedding_dim: int = 768
    batch_size: int = 32
    max_sequence_length: int = 512
    temperature: float = 0.7
    device: str = "auto"  # "cpu", "cuda", "auto"

    @classmethod
    def from_env(cls) -> 'DeepseekConfig':
        return cls(
            model_name=os.getenv("LLM_MODEL", os.getenv("EMBEDDING_MODEL", cls.model_name)),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", cls.batch_size)),
            max_sequence_length=int(os.getenv("LLM_MAX_TOKENS", cls.max_sequence_length)),
            temperature=float(os.getenv("LLM_TEMPERATURE", cls.temperature)),
            device=os.getenv("EMBEDDING_DEVICE", cls.device),
        )

    @staticmethod
    def list_supported_models() -> Dict[str, Dict]:
        """Return all supported models with their metadata."""
        return SUPPORTED_MODELS

    def validate_model(self) -> bool:
        """Check if the configured model is in the supported list."""
        if self.model_name not in SUPPORTED_MODELS:
            import logging
            logging.getLogger(__name__).warning(
                f"⚠️ Model '{self.model_name}' is not in the pre-tested list. "
                f"It may still work if it's a HuggingFace causal LM with chat template."
            )
            return False
        return True
