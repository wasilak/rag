import chromadb
import logging
from dotenv import load_dotenv
from libs.args import parse_arguments
from libs.database import process_data_fill
from libs.search import process_search
from libs.chat import process_chat
from chromadb.config import Settings

logger = logging.getLogger("RAG")


def main():
    args = parse_arguments()

    logging.basicConfig(level=args.log_level)

    client = chromadb.PersistentClient(path=args.db_path, settings=Settings(anonymized_telemetry=False))

    if args.subparser == "data-fill":
        process_data_fill(
            client=client,
            collection=args.collection,
            source_paths=args.source_path,
            source_type=args.source_type,
            mode=args.mode,
            cleanup=args.cleanup,
            llm=args.llm,
            embedding_model=args.embedding_model,
            embedding_llm=args.embedding_llm,
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
            embedding_llm=args.embedding_llm
        )

    elif args.subparser == "chat":
        process_chat(
            client=client,
            collection=args.collection,
            llm=args.llm,
            model=args.model,
            embedding_model=args.embedding_model,
            embedding_llm=args.embedding_llm,
        )


if __name__ == "__main__":
    load_dotenv()
    main()
