"""
SciBERT Embedding Service for Academic Papers.

This module provides functionality to generate SciBERT embeddings for academic papers
from JSON data files. It automatically detects the most recent input file and
provides configurable embedding generation.
"""

import torch
from typing import List, Optional

from models.configurators.LLMConfig import LLMConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time as _time

logger = logging.getLogger(__name__)


class LLMClient:
    """Service for generating SciBERT embeddings from academic papers."""

    def __init__(self, config: Optional[LLMConfig] = None, model_name = None):
        """Initialize the SciBERT embedding service.

        Args:
            config: SciBERT configuration. If None, loads from environment.
        """
        self.config = config or LLMConfig.from_env()
        self.tokenizer = None
        self.model = None
        self.model_name = model_name or self.config.model_name
        self.device = None
        
        # LLM call tracking
        self._llm_call_count = 0
        self._llm_call_log = []  # list of {purpose, timestamp, duration_ms}
        self._llm_total_time = 0.0  # cumulative seconds

        # Prometheus integration (optional — works without it)
        self._prometheus = None
        try:
            from clients.prometheus.PrometheusClient import get_prometheus_integration
            self._prometheus = get_prometheus_integration()
        except Exception:
            pass

        self._load_model()

    def _load_model(self):
        logger.info(f"Loading model: {self.model_name}")
        self.config.validate_model()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        self.model.eval()
        logger.info(f"✅ Model loaded: {self.model_name} on {self.model.device}")

    def reload_model(self, model_name: str) -> str:
        """Hot-swap the LLM model at runtime.
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            Status message
        """
        old_name = self.model_name
        logger.info(f"🔄 Switching LLM: {old_name} → {model_name}")
        
        # Free old model memory
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.config.model_name = model_name
        self._load_model()
        self.reset_llm_stats()
        
        return f"Model switched from {old_name} to {model_name}"

    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence-level embeddings."""
        token_embeddings = model_output.last_hidden_state  # (batch, seq_len, hidden)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(dim=1), min=1e-9
        )

    def generate_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding milvus as list of floats
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

    # Purpose-based token limits — keep outputs short for fast inference
    # on small models (1.5B).  Every extra token ≈ 0.2s on CPU / 0.02s on GPU.
    PURPOSE_TOKEN_LIMITS = {
        'routing':              32,   # just a label
        'template_selection':   24,   # single template key name
        'entity_extraction':    150,  # compact JSON dict
        'author_extraction':    64,   # list of names
        'cypher_generation':    128,  # Cypher query + params
        'claim_extraction':     128,  # short bullet list of claims
        'scifact_verification': 16,   # SUPPORTED / CONTRADICTED / NO_EVIDENCE
        'answer_synthesis':     200,  # main answer — concise, 3-5 sentences
        'general':              128,
    }

    def generate_content(self, prompt: str, system_prompt: str = None,
                         purpose: str = "general", max_tokens: int = None) -> str | None:
        """Generate text using the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            purpose: Label for tracking (e.g. 'routing', 'cypher_generation',
                     'answer_synthesis', 'scifact_verification', 'claim_extraction')
            max_tokens: Override max_new_tokens for this call. If None, uses
                        purpose-based default from PURPOSE_TOKEN_LIMITS.
        """
        call_start = _time.time()
        self._llm_call_count += 1
        call_number = self._llm_call_count
        
        effective_max_tokens = max_tokens or self.PURPOSE_TOKEN_LIMITS.get(
            purpose, self.config.max_sequence_length
        )
        logger.info(f"LLM call #{call_number} | purpose={purpose} | max_tokens={effective_max_tokens}")
        
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

            # Cap input length to leave room for generation
            max_input_tokens = self.config.max_sequence_length - effective_max_tokens
            max_input_tokens = max(max_input_tokens, 128)  # safety floor

            inputs = self.tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
            )

            device = self.model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=effective_max_tokens,
                    temperature=self.config.temperature or 0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[0, input_length:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            duration = _time.time() - call_start
            self._llm_total_time += duration
            self._llm_call_log.append({
                'call_number': call_number,
                'purpose': purpose,
                'duration_sec': round(duration, 3),
                'tokens_generated': len(generated_ids),
                'timestamp': _time.strftime('%H:%M:%S')
            })
            logger.info(f"LLM call #{call_number} completed | purpose={purpose} | {duration:.2f}s | {len(generated_ids)} tokens")

            # Push to Prometheus: AI call count, duration, tokens generated
            if self._prometheus:
                self._prometheus.record_ai_call(purpose, duration, len(generated_ids))

            return response

        except Exception as e:
            duration = _time.time() - call_start
            self._llm_total_time += duration
            self._llm_call_log.append({
                'call_number': call_number,
                'purpose': purpose,
                'duration_sec': round(duration, 3),
                'tokens_generated': 0,
                'error': str(e),
                'timestamp': _time.strftime('%H:%M:%S')
            })
            logger.error(f"LLM call #{call_number} FAILED | purpose={purpose} | {duration:.2f}s | {e}", exc_info=True)

            # Push failed call to Prometheus (0 tokens)
            if self._prometheus:
                self._prometheus.record_ai_call(purpose, duration, 0)

            return None
    
    def get_llm_stats(self) -> dict:
        """Get LLM usage statistics."""
        by_purpose = {}
        for entry in self._llm_call_log:
            p = entry['purpose']
            if p not in by_purpose:
                by_purpose[p] = {'count': 0, 'total_time': 0.0, 'errors': 0}
            by_purpose[p]['count'] += 1
            by_purpose[p]['total_time'] += entry['duration_sec']
            if 'error' in entry:
                by_purpose[p]['errors'] += 1
        
        for p in by_purpose:
            by_purpose[p]['avg_time'] = round(
                by_purpose[p]['total_time'] / max(1, by_purpose[p]['count']), 3
            )
            by_purpose[p]['total_time'] = round(by_purpose[p]['total_time'], 3)
        
        return {
            'total_llm_calls': self._llm_call_count,
            'total_llm_time_sec': round(self._llm_total_time, 3),
            'avg_time_per_call_sec': round(self._llm_total_time / max(1, self._llm_call_count), 3),
            'by_purpose': by_purpose,
            'recent_calls': self._llm_call_log[-10:]  # last 10 calls
        }
    
    def reset_llm_stats(self):
        """Reset LLM call statistics."""
        self._llm_call_count = 0
        self._llm_call_log.clear()
        self._llm_total_time = 0.0