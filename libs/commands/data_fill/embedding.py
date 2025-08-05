import argparse
import os
import logging
from chromadb.utils.embedding_functions import (
    OllamaEmbeddingFunction,
    GoogleGenerativeAiEmbeddingFunction,
    OpenAIEmbeddingFunction,
)
from chromadb.api.types import EmbeddingFunction
from ...models import get_best_model

logger = logging.getLogger("RAG")


def set_embedding_function(args: argparse.Namespace) -> EmbeddingFunction:
    """Set up the appropriate embedding function based on the LLM provider"""
    logger.debug("Setting embedding function")

    # Get validated embedding model
    # For embedding operations, use embedding_ollama_host:embedding_ollama_port
    validated_model = get_best_model(
        args.embedding_llm,
        args.embedding_ollama_host,
        args.embedding_ollama_port,
        args.model,
        "embedding",
    )

    if args.embedding_llm == "ollama":
        logger.debug(f"Using Ollama embedding model '{validated_model}'")
        return OllamaEmbeddingFunction(
            url=f"http://{args.embedding_ollama_host}:{args.embedding_ollama_port}",
            model_name=validated_model,
        )
    elif args.embedding_llm == "openai":
        logger.debug(f"Using OpenAI embedding model '{validated_model}'")
        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=validated_model,
        )
    elif args.embedding_llm == "gemini":
        logger.debug(f"Using Gemini embedding model '{validated_model}'")
        return GoogleGenerativeAiEmbeddingFunction(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name=f"models/{validated_model}",
        )
    else:
        logger.error(f"Invalid embedding function provider: {args.embedding_llm}")
        exit(1)
