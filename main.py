import chromadb
import logging
from dotenv import load_dotenv
from libs.args import parse_arguments
from libs.commands.data_fill.data import process_data_fill
from libs.commands.search.search import process_search
from libs.commands.chat.chat import process_chat
from libs.commands.web.web import process_web
from libs.list_models import process_list_models
from libs.cache import pre_cache_llm_models, CacheRequirements
from chromadb.config import Settings
from libs.utils import setup_colored_logging, validate_client_and_exit

logger = logging.getLogger("RAG")


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Determine if Chroma should be used for data-fill
    insert_into_chroma = True
    if getattr(args, "subparser", None) == "data-fill":
        insert_into_chroma = getattr(args, "insert_into_chroma", True)

    # Set up colored logging
    setup_colored_logging(args.log_level)

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

        # Initialize ChromaDB client - use persistent client if path provided, otherwise HTTP client
        if len(args.chromadb_path) > 0:
            client = chromadb.PersistentClient(
                path=args.chromadb_path, settings=chroma_settings
            )
        else:
            # https://docs.trychroma.com/reference/python/client
            client = chromadb.HttpClient(
                host=args.chromadb_host, port=args.chromadb_port, settings=chroma_settings
            )

    # Route to appropriate command handler based on subparser
    if args.subparser == "data-fill":
        process_data_fill(client=client, args=args)

    elif args.subparser == "search":
        # Validate that we have a ChromaDB client before proceeding
        if not validate_client_and_exit(client, "perform search", logger):
            return
        assert client is not None  # Type hint for pyright
        process_search(client=client, args=args)

    elif args.subparser == "chat":
        # Validate that we have a ChromaDB client before proceeding
        if not validate_client_and_exit(client, "start chat", logger):
            return
        assert client is not None  # Type hint for pyright
        process_chat(client=client, args=args)

    elif args.subparser == "web":
        # Validate that we have a ChromaDB client before proceeding
        if not validate_client_and_exit(client, "start web interface", logger):
            return
        assert client is not None  # Type hint for pyright
        process_web(client=client, args=args)


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    main()
