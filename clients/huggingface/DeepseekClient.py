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
        """Load SciBERT model and tokenizer."""
        print(f"Loading Deepseek model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(self.config.model_name)

        # Determine device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Model loaded successfully on device: {self.device}")

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

    def generate_content(self, prompt):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_sequence_length,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return type('Response', (), {'text': response_text.strip()})()

        except Exception as e:
            logger.error(f"DeepseekClient generation error: {e}")
            return None