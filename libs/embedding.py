import os
import logging
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction, GoogleGenerativeAiEmbeddingFunction, OpenAIEmbeddingFunction

logger = logging.getLogger("RAG")


def set_embedding_function(function, model):
    """Set up the appropriate embedding function based on the LLM provider"""
    logger.info(f"Setting embedding function")
    if function == "ollama":
        logger.info(f"Using Ollama embedding model '{model}'")
        return OllamaEmbeddingFunction(
            url="http://localhost:11434",
            model_name=model,
        )
    elif function == "openai":
        logger.info(f"Using OpenAI embedding model '{model}'")
        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif function == "gemini":
        logger.info(f"Using Gemini embedding model")
        return GoogleGenerativeAiEmbeddingFunction(
            api_key=os.getenv("GEMINI_API_KEY"),
            # model_name= "models/embedding-001",
        )
    else:
        logger.error(f"Invalid embedding function: {function}")
        exit(1) 
