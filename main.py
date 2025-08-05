import chromadb
import logging
import colorlog
from dotenv import load_dotenv
from libs.args import parse_arguments
from libs.commands.data_fill.data import process_data_fill
from libs.search import process_search
from libs.chat import process_chat
from libs.web import process_web
from libs.list_models import process_list_models
from libs.cache import pre_cache_llm_models, CacheRequirements
from chromadb.config import Settings

logger = logging.getLogger("RAG")


def main():
    args = parse_arguments()
    # Determine if Chroma should be used for data-fill
    insert_into_chroma = True
    if getattr(args, "subparser", None) == "data-fill":
        insert_into_chroma = getattr(args, "insert_into_chroma", True)

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
        process_list_models(args=args, force_refresh=True)
        return

    # Pre-cache models that will be needed based on the LLM provider being used
    llm_provider = getattr(args, 'llm', 'ollama')  # Default to ollama if not specified
    embedding_llm_provider = getattr(args, 'embedding_llm', 'ollama')  # Default to ollama if not specified

    # Only pre-cache models and set up ChromaDB client if not skipping Chroma for data-fill
    chroma_settings = Settings(anonymized_telemetry=False)
    client = None

    if args.subparser != "data-fill" or insert_into_chroma:
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
            args=args,
        )

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

        process_data_fill(client=client, args=args)

    elif args.subparser == "search":
        if client is None:
            logger.error("ChromaDB client is not initialized. Cannot perform search.")
            return

        process_search(client=client, args=args)

    elif args.subparser == "chat":
        if client is None:
            logger.error("ChromaDB client is not initialized. Cannot start chat.")
            return
        process_chat(client=client, args=args)

    elif args.subparser == "web":
        if client is None:
            logger.error("ChromaDB client is not initialized. Cannot start web interface.")
            return
        process_web(client=client, args=args)


if __name__ == "__main__":
    load_dotenv()
    main()
