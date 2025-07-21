import os
import logging
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction, GoogleGenerativeAiEmbeddingFunction, OpenAIEmbeddingFunction

logger = logging.getLogger("RAG")


def set_embedding_function(function_provider, model):
    """Set up the appropriate embedding function based on the LLM provider"""
    logger.debug("Setting embedding function")
    if function_provider == "ollama":
        logger.debug(f"Using Ollama embedding model '{model}'")
        return OllamaEmbeddingFunction(
            url="http://localhost:11434",
            model_name=model,
        )
    elif function_provider == "openai":
        logger.debug(f"Using OpenAI embedding model '{model}'")
        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif function_provider == "gemini":
        logger.debug("Using Gemini embedding model")
        return GoogleGenerativeAiEmbeddingFunction(
            api_key=os.getenv("GEMINI_API_KEY"),
            # model_name= "models/embedding-001",
        )
    else:
        logger.error(f"Invalid embedding function provider: {function_provider}")
        exit(1)
