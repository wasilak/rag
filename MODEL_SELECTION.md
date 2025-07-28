# Model Selection System

This document describes the model selection and validation system for the RAG application, which supports three LLM providers: Ollama, OpenAI, and Gemini.

## Overview

The model selection system provides:

- **Default models** for each provider (chat and embedding)
- **Model validation** to ensure requested models exist
- **Automatic fallback** to default models if validation fails
- **Model listing** functionality to see available models

## Default Models

The system includes pre-configured default models for each provider:

### Ollama

- **Chat Model**: `qwen3:8b`
- **Embedding Model**: `nomic-embed-text`
- **Document Cleaning Model**: `qwen3:8b` (same as chat model)

### OpenAI

- **Chat Model**: `gpt-4o`
- **Embedding Model**: `text-embedding-3-small`
- **Document Cleaning Model**: `gpt-4o` (same as chat model)

### Gemini

- **Chat Model**: `gemini-1.5-flash`
- **Embedding Model**: `text-embedding-004`
- **Document Cleaning Model**: `gemini-1.5-flash` (same as chat model)

## Listing Available Models

You can list all available models for a specific provider using the `list-models` command:

```bash
# List Ollama models
python main.py list-models ollama

# List OpenAI models
python main.py list-models openai

# List Gemini models
python main.py list-models gemini
```

The output shows model names, sizes, creation dates, and other relevant information in a formatted table.

## Model Validation

When you specify a model, the system automatically:

1. **Checks if the model exists** by querying the provider's API
2. **Validates model capabilities** (if supported by the provider)
3. **Falls back to default** if the specified model is not found
4. **Logs warnings** when fallback occurs

### Ollama Model Matching

For Ollama models, the system handles tag formats intelligently:

- If you specify `nomic-embed-text`, it will match `nomic-embed-text:latest`
- Both the full name (`model:tag`) and base name (`model`) are supported

## Usage Examples

### Using Default Models

```bash
# Uses default models for each provider
python main.py search "your query" --llm ollama
python main.py chat --llm openai
```

### Specifying Custom Models

```bash
# Use a specific chat model
python main.py search "your query" --llm ollama --model "deepseek-r1:14b"

# Use custom models for both chat and embedding
python main.py search "your query" \
  --llm openai \
  --model "gpt-4o-mini" \
  --embedding-llm openai \
  --embedding-model "text-embedding-3-large"

# Use custom models for document cleaning
python main.py data-fill https://example.com --source-type url \
  --enable-cleaning \
  --cleaning-llm openai \
  --cleaning-model "gpt-4o-mini"
```

### Model Validation in Action

```bash
# This will validate the model and use it if found
python main.py search "test" --model "qwen3:8b"

# This will warn and fall back to default (qwen3:8b)
python main.py search "test" --model "nonexistent-model"
```

## Environment Variables

You can set default models via environment variables:

```bash
export RAG_LLM=openai
export RAG_MODEL=gpt-4o-mini
export RAG_EMBEDDING_LLM=openai
export RAG_EMBEDDING_MODEL=text-embedding-3-small
export RAG_CLEANING_LLM=openai
export RAG_CLEANING_MODEL=gpt-4o-mini
```

## API Keys Required

Make sure to set the appropriate API keys in your environment:

```bash
# For OpenAI
export OPENAI_API_KEY=your_openai_api_key

# For Gemini
export GEMINI_API_KEY=your_gemini_api_key

# Ollama runs locally, no API key needed
```

## Model Manager API

The system provides a programmatic API for model management:

### Basic Functions

```python
from libs.models import get_model_manager, get_best_model

# Get model manager instance
manager = get_model_manager()

# List models for a provider
models = manager.list_models("ollama")

# Get default model
default_chat = manager.get_default_model("ollama", "chat")

# Validate a specific model
is_valid = manager.validate_model("ollama", "qwen3:8b", "chat")

# Get best available model (with fallback)
best_model = get_best_model("ollama", "custom-model", "chat")
```

### Convenience Functions

```python
from libs.models import list_provider_models, validate_model_choice, get_best_model

# List all models for a provider
ollama_models = list_provider_models("ollama")

# Validate a model choice
is_valid = validate_model_choice("openai", "gpt-4o", "chat")

# Get the best available model (validates and falls back if needed)
model = get_best_model("gemini", "custom-model", "embedding")
```

## Troubleshooting

### Common Issues

1. **"Model not found" warnings**: The requested model doesn't exist for the provider. The system will fall back to the default model.

2. **API key errors**: Make sure you have the correct API keys set in your environment variables.

3. **Ollama connection issues**: Ensure Ollama is running locally on `http://localhost:11434`.

4. **Network timeouts**: Model listing requires API calls. Check your internet connection and API quotas.

### Debug Mode

Use debug logging to see detailed model validation information:

```bash
python main.py --log-level DEBUG search "your query"
python main.py --log-level DEBUG data-fill https://example.com --source-type url --enable-cleaning
```

This will show:

- Model validation attempts
- API responses
- Fallback decisions
- Model selection reasoning
- Document cleaning progress and character reduction statistics

## Model Types

The system recognizes three model types:

- **chat**: Models used for text generation and conversation
- **embedding**: Models used for text embedding and semantic search
- **cleaning**: Models used for document cleaning (removes ads, navigation, obsolete content)

Note: Document cleaning uses the same models as chat models, as it requires text generation capabilities to clean and restructure content.

Each provider may have different models optimized for these tasks.

## Future Enhancements

Planned improvements include:

- Model performance caching
- Custom model configuration files
- Model recommendation based on task type
- Integration with model registries
