import os
import logging
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction, GoogleGenerativeAiEmbeddingFunction, OpenAIEmbeddingFunction

logger = logging.getLogger("RAG")


def set_embedding_function(function_provider, model):
    """Set up the appropriate embedding function based on the LLM provider"""
    logger.info("Setting embedding function")
    if function_provider == "ollama":
        logger.info(f"Using Ollama embedding model '{model}'")
        return OllamaEmbeddingFunction(
            url="http://localhost:11434",
            model_name=model,
        )
    elif function_provider == "openai":
        logger.info(f"Using OpenAI embedding model '{model}'")
        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif function_provider == "gemini":
        logger.info("Using Gemini embedding model")
        return GoogleGenerativeAiEmbeddingFunction(
            api_key=os.getenv("GEMINI_API_KEY"),
            # model_name= "models/embedding-001",
        )
    else:
        logger.error(f"Invalid embedding function provider: {function_provider}")
        exit(1)
