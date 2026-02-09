"""
SciBERT Embedding Service for Academic Papers.

This module provides functionality to generate SciBERT embeddings for academic papers
from JSON data files. It automatically detects the most recent input file and
provides configurable embedding generation.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional

from models.configurators.DeepseekConfig import DeepseekConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)


class DeepseekClient:
    """Service for generating SciBERT embeddings from academic papers."""

    def __init__(self, config: Optional[DeepseekConfig] = None):
        """Initialize the SciBERT embedding service.

        Args:
            config: SciBERT configuration. If None, loads from environment.
        """
        self.config = config or DeepseekConfig.from_env()
        self.tokenizer = None
        self.model = None
        self.device = None
        self._load_model()

    def _load_model(self):
        print(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # This is the important change
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"   # or "cuda" / "cpu"
        )

        self.model.eval()
        print(f"✅ Model loaded on device: {self.model.device}")

    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence-level embeddings."""
        token_embeddings = model_output.last_hidden_state  # (batch, seq_len, hidden)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(dim=1), min=1e-9
        )

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_sequence_length
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = self._mean_pooling(outputs, inputs["attention_mask"])
        return embedding.squeeze().cpu().tolist()

    def generate_content(self, prompt: str, system_prompt: str = None) -> str | None:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            # Critical fix
            device = self.model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Optional: add this temporarily to confirm
            print("Model device:", device)
            print("input_ids device:", inputs["input_ids"].device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_sequence_length,
                    temperature=self.config.temperature or 0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[0, input_length:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return None