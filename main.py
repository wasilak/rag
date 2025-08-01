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
from libs.cache import pre_cache_llm_models, CacheRequirements
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

    # Set up caching requirements
    cache_requirements = CacheRequirements(
        llm_provider=llm_provider,
        embedding_llm_provider=embedding_llm_provider,
        subcommand=args.subparser
    )

    # Pre-cache models
    pre_cache_llm_models(
        requirements=cache_requirements,
        process_list_models=process_list_models,
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        embedding_ollama_host=args.embedding_ollama_host,
        embedding_ollama_port=args.embedding_ollama_port
    )

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
            browser=args.browser,
            cors_origins=args.cors_origins,
            secret_key=args.secret_key,
            max_history=args.max_history,
            timeout=args.timeout,
            workers=args.workers,
        )


if __name__ == "__main__":
    load_dotenv()
    main()
