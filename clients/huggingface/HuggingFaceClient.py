from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

logger = logging.getLogger(__name__)
class HuggingFaceClient:
    def __init__(self, model_name, temperature, max_tokens, token=None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(f"Loading Hugging Face model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=token,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_content(self, prompt):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return type('Response', (), {'text': response_text.strip()})()

        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            return None