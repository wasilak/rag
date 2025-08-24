import logging
import argparse
from chromadb.api import ClientAPI

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chromadb.api.models.Collection import Collection

from .embedding import set_embedding_function
from .documents_types.markdown import process_markdown_documents

logger = logging.getLogger("RAG")


def delete_collection(client: ClientAPI, collection: str) -> None:
    """Delete a collection from ChromaDB"""
    logger.debug(f"Deleting collection '{collection}'")
    try:
        client.delete_collection(name=collection)
        logger.debug(f"Collection '{collection}' deleted")
    except Exception as e:
        print(f"Error deleting collection {collection}: {e}")


def create_get_collection(
    args: argparse.Namespace,
    client: ClientAPI, collection_name: str
) -> None:
    embedding_function = set_embedding_function(args)

    try:
        logger.debug(
            f"Creating/getting collection {args.collection_name} with Ollama embedding function..."
        )
        collection = client.get_or_create_collection(name=args.collection_name, embedding_function=embedding_function)
        logger.debug(f"Collection '{collection}' created/gotten")
    except Exception as e:
        logger.error(f"Error creating/getting collection {args.collection_name}: {e}")
        exit(1)


def insert_into_collection(
    collection: Collection,
    raw_documents,
    args: argparse.Namespace,
) -> None:
    logger.debug(
        f"Bootstrapping collection '{args.collection_name}' with {len(raw_documents)} documents"
    )
    logger.debug(
        f"Using embedding model '{args.embedding_model}' with provider '{args.embedding_llm}'"
    )

    logger.debug(f"Splitting {len(raw_documents)} documents into chunks")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(raw_documents)
    logger.debug(
        f"Split {len(raw_documents)} documents into {len(chunks)} chunks"
    )

    documents, metadata, ids = process_markdown_documents(chunks, args.mode, args.id_prefix)
    logger.debug(
        f"Upserting {len(documents)} documents into collection '{args.collection_name}'"
    )
    collection.upsert(documents=documents, metadatas=metadata, ids=ids)
    logger.debug(
        f"Upserted {len(documents)} documents into collection '{args.collection_name}'"
    )
