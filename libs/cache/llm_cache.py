"""LLM caching functionality."""

import argparse
import logging
from typing import Set, NamedTuple
from typing_extensions import TypeAlias

logger = logging.getLogger("RAG")

ProvidersToCacheSet: TypeAlias = Set[str]


class CacheRequirements(NamedTuple):
    """Requirements for caching LLM models."""
    llm_provider: str
    embedding_llm_provider: str
    subcommand: str


def get_providers_to_cache(requirements: CacheRequirements) -> ProvidersToCacheSet:
    """Get set of providers that need to be cached based on subcommand requirements.

    Args:
        requirements: Cache requirements including providers and subcommand.

    Returns:
        Set of provider names that need to be cached.
    """
    llm_provider = requirements.llm_provider
    embedding_llm_provider = requirements.embedding_llm_provider
    subcommand = requirements.subcommand

    # Pre-cache models based on the subcommand requirements
    if subcommand == "data-fill":
        # data-fill only needs embedding models
        return {embedding_llm_provider}
    elif subcommand in ["web", "search", "chat"]:
        # These commands need both LLM and embedding models
        return {llm_provider, embedding_llm_provider}
    else:
        # Default: cache both to be safe
        return {llm_provider, embedding_llm_provider}


def pre_cache_llm_models(
    requirements: CacheRequirements,
    process_list_models,  # Cannot type this due to circular import
    args: argparse.Namespace,
) -> None:
    """Pre-cache LLM models based on command requirements.

    Args:
        requirements: Cache requirements including providers and subcommand
        process_list_models: Function to list available models for a provider
        ollama_host: Ollama server host for LLM
        ollama_port: Ollama server port for LLM
        embedding_ollama_host: Ollama server host for embeddings
        embedding_ollama_port: Ollama server port for embeddings
    """
    providers_to_cache = get_providers_to_cache(requirements)

    for provider in providers_to_cache:
        logger.debug(f"Pre-caching models for {provider}...")
        try:
            # Use appropriate Ollama host/port based on subcommand and provider
            if provider == "ollama":
                if requirements.subcommand == "data-fill":
                    # data-fill only uses Ollama for embeddings
                    host, port = args.embedding_ollama_host, args.embedding_ollama_port
                    logger.debug(f"data-fill with Ollama: using embedding settings {host}:{port}")

                elif (requirements.llm_provider == "ollama" and requirements.embedding_llm_provider == "ollama"):
                    # Both LLM and embedding use Ollama - check if they use different instances
                    if (args.ollama_host == args.embedding_ollama_host and args.ollama_port == args.embedding_ollama_port):
                        # Same Ollama instance for both - use LLM settings
                        host, port = args.ollama_host, args.ollama_port
                        logger.debug(
                            "Ollama for both LLM and embedding (same instance): "
                            f"using LLM settings {host}:{port}"
                        )
                    else:
                        # Different Ollama instances - use embedding settings
                        # since that's what most operations need
                        host, port = args.embedding_ollama_host, args.embedding_ollama_port
                        logger.debug(
                            "Ollama for both LLM and embedding (different instances): "
                            f"using embedding settings {host}:{port}"
                        )
                elif requirements.llm_provider == "ollama":
                    # Only LLM uses Ollama
                    host, port = args.ollama_host, args.ollama_port
                    logger.debug(f"Ollama for LLM only: using {host}:{port}")
                else:
                    # Only embedding uses Ollama (or this is embedding pre-caching)
                    host, port = args.embedding_ollama_host, args.embedding_ollama_port
                    logger.debug(f"Ollama for embedding: using {host}:{port}")
            else:
                # For non-Ollama providers (gemini, openai), use embedding Ollama settings
                # since they might need Ollama for embeddings
                host, port = args.embedding_ollama_host, args.embedding_ollama_port
                logger.debug(f"Non-Ollama provider {provider}: using embedding Ollama {host}:{port}")

            process_list_models(
                provider=provider,
                force_refresh=False,
                silent=True,
                ollama_host=host,
                ollama_port=port
            )
        except Exception as e:
            logger.warning(f"Could not pre-cache models for {provider}: {e}")
            # Continue anyway - this is just optimization
