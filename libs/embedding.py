import os
import logging
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction, GoogleGenerativeAiEmbeddingFunction, OpenAIEmbeddingFunction
from chromadb.api.types import EmbeddingFunction
from .models import get_best_model

logger = logging.getLogger("RAG")


def set_embedding_function(function_provider: str, model: str, embedding_ollama_host:str, embedding_ollama_port: int) -> EmbeddingFunction:
    """Set up the appropriate embedding function based on the LLM provider"""
    logger.debug("Setting embedding function")

    # Get validated embedding model
    validated_model = get_best_model(function_provider, embedding_ollama_host, embedding_ollama_port, model, "embedding")

    if function_provider == "ollama":
        logger.debug(f"Using Ollama embedding model '{validated_model}'")
        return OllamaEmbeddingFunction(
          url=f"http://{embedding_ollama_host}:{embedding_ollama_port}",
            model_name=validated_model,
        )
    elif function_provider == "openai":
        logger.debug(f"Using OpenAI embedding model '{validated_model}'")
        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=validated_model,
        )
    elif function_provider == "gemini":
        logger.debug(f"Using Gemini embedding model '{validated_model}'")
        return GoogleGenerativeAiEmbeddingFunction(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name=f"models/{validated_model}",
        )
    else:
        logger.error(f"Invalid embedding function provider: {function_provider}")
        exit(1)
