# Llama Configuration Options

## Environment Variables

You can configure the Llama integration using the following environment variables:

### Provider Selection
- `LLAMA_PROVIDER`: Choose between 'ollama' (local) or 'huggingface' (cloud/local HF models)
  - Default: 'ollama'

### Model Configuration
- `LLAMA_MODEL`: Model name to use
  - For Ollama: 'llama2', 'mistral', 'codellama', etc.
  - For HuggingFace: 'meta-llama/Llama-2-7b-chat-hf', 'microsoft/DialoGPT-medium', etc.
  - Default: 'llama2'

### Generation Parameters
- `LLAMA_TEMPERATURE`: Temperature for text generation (0.0-1.0)
  - Default: '0.1'
- `LLAMA_MAX_TOKENS`: Maximum tokens to generate
  - Default: '1000'

### Ollama Configuration (for local models)
- `OLLAMA_BASE_URL`: Base URL for Ollama server
  - Default: 'http://localhost:11434'

### HuggingFace Configuration
- `HUGGINGFACE_TOKEN`: Your HuggingFace access token (for private models)
  - Optional, only needed for private/gated models

## Quick Start with Ollama (Recommended)

1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull llama2`
3. Start Ollama server: `ollama serve`
4. Set environment variables:
   ```bash
   export LLAMA_PROVIDER=ollama
   export LLAMA_MODEL=llama2
   ```

## Quick Start with HuggingFace

1. Set environment variables:
   ```bash
   export LLAMA_PROVIDER=huggingface
   export LLAMA_MODEL=microsoft/DialoGPT-medium
   export HUGGINGFACE_TOKEN=your_token_here  # if needed
   ```

Note: HuggingFace models will be downloaded locally and may require significant disk space and memory.

## Recommended Models

### Ollama (Local, Fast)
- `llama2` - Good general purpose model
- `mistral` - Fast and efficient
- `codellama` - Good for technical/scientific content

### HuggingFace (More variety)
- `microsoft/DialoGPT-medium` - Good conversational model, relatively small
- `meta-llama/Llama-2-7b-chat-hf` - High quality but requires more resources
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - Very small and fast