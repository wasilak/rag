import chromadb
import logging
import colorlog
from dotenv import load_dotenv
from libs.args import parse_arguments
from libs.database import process_data_fill
from libs.search import process_search
from libs.chat import process_chat
from libs.web import process_web
from libs.list_models import process_list_models
from chromadb.config import Settings

logger = logging.getLogger("RAG")


def main():
    args = parse_arguments()

    # Set up colored logging
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s:%(name)s:%(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(args.log_level)

    # Suppress noisy HTTP and library logs
    for noisy_logger in [
        "httpx",
        "urllib3",
        "chromadb",
        "openai",
        "httpcore",
        "boto3",
        "botocore",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Handle list-models command (doesn't need ChromaDB client)
    if args.subparser == "list-models":
        process_list_models(
            provider=args.provider,
            force_refresh=True,
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port
        )
        return

    # Pre-cache models that will be needed based on the LLM provider being used
    llm_provider = getattr(args, 'llm', 'ollama')  # Default to ollama if not specified
    embedding_llm_provider = getattr(args, 'embedding_llm', 'ollama')  # Default to ollama if not specified

    # Pre-cache models based on the subcommand requirements
    if args.subparser == "data-fill":
        # data-fill only needs embedding models
        providers_to_cache = {embedding_llm_provider}
    elif args.subparser == "web":
        # web needs both LLM and embedding models
        providers_to_cache = {llm_provider, embedding_llm_provider}
    elif args.subparser == "search":
        # search needs both LLM and embedding models
        providers_to_cache = {llm_provider, embedding_llm_provider}
    elif args.subparser == "chat":
        # chat needs both LLM and embedding models
        providers_to_cache = {llm_provider, embedding_llm_provider}
    else:
        # Default: cache both
        providers_to_cache = {llm_provider, embedding_llm_provider}

    for provider in providers_to_cache:
        logger.debug(f"Pre-caching models for {provider}...")
        try:
            # Use appropriate Ollama host/port based on subcommand and provider
            if provider == "ollama":
                if args.subparser == "data-fill":
                    # data-fill only uses Ollama for embeddings
                    host, port = args.embedding_ollama_host, args.embedding_ollama_port
                    logger.debug(f"data-fill with Ollama: using embedding settings {host}:{port}")
                elif llm_provider == "ollama" and embedding_llm_provider == "ollama":
                    # Both LLM and embedding use Ollama - check if they use different instances
                    if (args.ollama_host == args.embedding_ollama_host
                            and args.ollama_port == args.embedding_ollama_port):
                        # Same Ollama instance for both - use LLM settings
                        host, port = args.ollama_host, args.ollama_port
                        logger.debug(f"Ollama for both LLM and embedding (same instance): using LLM settings {host}:{port}")
                    else:
                        # Different Ollama instances - this is tricky, use embedding settings for pre-caching
                        # since that's what most operations need
                        host, port = args.embedding_ollama_host, args.embedding_ollama_port
                        logger.debug(f"Ollama for both LLM and embedding (different instances): using embedding settings {host}:{port}")
                elif llm_provider == "ollama":
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

    chroma_settings = Settings(anonymized_telemetry=False)

    if len(args.chromadb_path) > 0:
        client = chromadb.PersistentClient(
            path=args.chromadb_path, settings=chroma_settings
        )
    else:
        # https://docs.trychroma.com/reference/python/client
        client = chromadb.HttpClient(
            host=args.chromadb_host, port=args.chromadb_port, settings=chroma_settings
        )

    if args.subparser == "data-fill":
        if args.cleanup:
            logger.info("Cleanup enabled: collection will be deleted before filling.")
        if args.clean_content:
            logger.info(
                "Document cleaning enabled: HTML tags and UI elements will be removed before Markdown conversion."
            )
        else:
            logger.info(
                "Document cleaning disabled: raw HTML will be converted to Markdown without pre-cleaning."
            )
        if args.extract_wisdom:
            logger.info(
                f"Wisdom extraction enabled: {args.fabric_command} will be used to extract key insights."
            )
        process_data_fill(
            client=client,
            collection_name=args.collection,
            source_paths=args.source_path,
            mode=args.mode,
            cleanup=args.cleanup,
            embedding_model=args.embedding_model,
            embedding_llm=args.embedding_llm,
            embedding_ollama_host=args.embedding_ollama_host,
            embedding_ollama_port=args.embedding_ollama_port,
            bucket_name=args.bucket_name,
            bucket_path=args.bucket_path,
            clean_content=args.clean_content,
            enable_wisdom=args.extract_wisdom,
            fabric_command=args.fabric_command,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

    elif args.subparser == "search":
        process_search(
            client=client,
            collection=args.collection,
            query=args.query,
            llm=args.llm,
            model=args.model,
            dry_run=args.dry_run,
            embedding_model=args.embedding_model,
            embedding_llm=args.embedding_llm,
            embedding_ollama_host=args.embedding_ollama_host,
            embedding_ollama_port=args.embedding_ollama_port,
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
        )

    elif args.subparser == "chat":
        process_chat(
            client=client,
            collection=args.collection,
            llm=args.llm,
            model=args.model,
            embedding_model=args.embedding_model,
            embedding_llm=args.embedding_llm,
            embedding_ollama_host=args.embedding_ollama_host,
            embedding_ollama_port=args.embedding_ollama_port,
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
        )

    elif args.subparser == "web":
        process_web(
            client=client,
            collection=args.collection,
            llm=args.llm,
            model=args.model,
            embedding_model=args.embedding_model,
            embedding_llm=args.embedding_llm,
            embedding_ollama_host=args.embedding_ollama_host,
            embedding_ollama_port=args.embedding_ollama_port,
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
            port=args.port,
            host=args.host,
            debug=args.debug,
            no_browser=args.no_browser,
            cors_origins=args.cors_origins,
            secret_key=args.secret_key,
            max_history=args.max_history,
            timeout=args.timeout,
            workers=args.workers,
        )


if __name__ == "__main__":
    load_dotenv()
    main()
