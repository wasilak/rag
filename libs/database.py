import logging
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter

from libs.s3 import upload_markdown_to_s3
from .validation import validate_s3_bucket_name, validate_s3_bucket_path
from .embedding import set_embedding_function
from chromadb.api import ClientAPI
from typing import Sequence, List, Optional
from langchain_core.documents import Document
from chromadb.api.types import (
    Metadata,
    QueryResult,
    OneOrMany,
)

from .document_loader import load_documents as load_docs

logger = logging.getLogger("RAG")


def delete_collection(client: ClientAPI, collection: str) -> None:
    """Delete a collection from ChromaDB"""
    logger.debug(f"Deleting collection '{collection}'")
    try:
        client.delete_collection(name=collection)
        logger.debug(f"Collection '{collection}' deleted")
    except Exception as e:
        print(f"Error deleting collection {collection}: {e}")


def load_documents(
    source_path: str,
    mode: str,
    clean_content: bool = False,
    enable_wisdom: bool = False,
    fabric_command: str = "fabric",
) -> list[Document] | Sequence[Document]:
    """Load documents from file or URL"""
    return load_docs(
        source_path,
        mode,
        clean_content=clean_content,
        enable_wisdom=enable_wisdom,
        fabric_command=fabric_command,
    )


def process_markdown_documents(
    chunks: Sequence[Document], mode: str, id_prefix: str
) -> tuple[List[str], Optional[OneOrMany[Metadata]], List[str]]:
    """Process markdown documents and extract metadata"""
    documents: List[str] = []
    metadata: Optional[OneOrMany[Metadata]] = []
    ids: List[str] = []
    logger.debug(f"Processing {len(chunks)} chunks in {mode} mode")

    if mode == "single":
        i = 0
        for chunk in chunks:
            documents.append(chunk.page_content)
            ids.append(id_prefix + "_" + str(i))
            metadata.append(chunk.metadata)
            i += 1

    if mode == "elements":
        # Build element_id map
        elements_by_id = {doc.metadata["element_id"]: doc for doc in chunks}

        def get_ancestor_chain(doc_id, elements_by_id, max_levels=100):
            chain = []
            current_id = doc_id
            levels = 0
            while levels < max_levels:
                doc = elements_by_id.get(current_id)
                if not doc or "parent_id" not in doc.metadata:
                    break
                parent_id = doc.metadata["parent_id"]
                parent = elements_by_id.get(parent_id)
                if not parent:
                    break
                chain.insert(0, parent)
                current_id = parent_id
                levels += 1
            return chain

        for i, chunk in enumerate(chunks):
            element_id = chunk.metadata["element_id"]
            parent_chain_docs = get_ancestor_chain(element_id, elements_by_id)

            documents.append(chunk.page_content)
            ids.append(f"ID_{i}")

            temp_metadata = {}
            for key, value in chunk.metadata.items():
                temp_metadata[key] = (
                    ", ".join(value) if isinstance(value, list) else value
                )

            if parent_chain_docs:
                temp_metadata["page_title"] = parent_chain_docs[0].page_content
                temp_metadata["resolved_title"] = (
                    chunk.page_content
                    if chunk.metadata.get("category") == "Title"
                    else parent_chain_docs[-1].page_content
                )
            else:
                temp_metadata["page_title"] = chunk.page_content
                temp_metadata["resolved_title"] = chunk.page_content

            metadata.append(temp_metadata)

    return documents, metadata, ids


def bootstrap_db(
    client: ClientAPI,
    collection_name: str,
    raw_documents,
    embedding_model: str,
    embedding_llm: str,
    embedding_ollama_host: str,
    embedding_ollama_port: int,
    mode: str,
    id_prefix: str,
    bucket_name: str,
    bucket_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Bootstrap the database with documents"""
    logger.debug(
        f"Bootstrapping collection '{collection_name}' with {
            len(raw_documents)} documents"
    )
    logger.debug(
        f"Using embedding model '{
            embedding_model}' with provider '{embedding_llm}'"
    )

    # LLM cleanup removed: documents are now cleaned only via HTML/Markdown pre-processing

    # Upload to S3 if enabled (after cleaning, before chunking)
    if bucket_name:
        if not validate_s3_bucket_name(bucket_name):
            logger.error(f"Invalid S3 bucket name: {bucket_name}")
            return
        if not validate_s3_bucket_path(bucket_path):
            logger.error(f"Invalid S3 bucket path: {bucket_path}")
            return
        logger.info("Uploading documents to S3")
        for doc in raw_documents:
            base_title = doc.metadata.get("base_title")
            # Add bucket_path to metadata
            doc.metadata["bucket_path"] = bucket_path
            if doc.metadata.get("is_wisdom") or not doc.metadata.get("is_original"):
                # Regular files (non-original) and wisdom files go to bucket_path
                file_title = base_title
                file_path = (
                    file_title + ".md"
                    if not bucket_path
                    else f"{bucket_path}/{file_title}.md"
                )
                upload_path = bucket_path
            else:
                # Original files only go to original/ when they have a wisdom counterpart
                file_title = f"{base_title}_original"
                file_path = (
                    f"original/{file_title}.md"
                    if not bucket_path
                    else f"{bucket_path}/original/{file_title}.md"
                )
                upload_path = (
                    "original"
                    if not bucket_path
                    else f"{
                        bucket_path}/original"
                )
            doc.metadata["file_path"] = file_path
            upload_markdown_to_s3(
                doc.page_content, file_title, upload_path, bucket_name
            )
        logger.debug("S3 upload completed")

    embedding_function = set_embedding_function(
        embedding_llm, embedding_model, embedding_ollama_host, embedding_ollama_port
    )

    try:
        logger.debug(
            f"Creating/getting collection {
                collection_name} with Ollama embedding function..."
        )
        collection = client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_function
        )
        logger.debug(f"Collection '{collection}' created/gotten")
    except Exception as e:
        logger.error(f"Error creating/getting collection {collection_name}: {e}")
        exit(1)

    logger.debug(f"Splitting {len(raw_documents)} documents into chunks")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(raw_documents)
    logger.debug(
        f"Split {len(raw_documents)} documents into {
                 len(chunks)} chunks"
    )

    documents, metadata, ids = process_markdown_documents(chunks, mode, id_prefix)

    logger.debug(
        f"Upserting {len(documents)} documents into collection '{
            collection_name}'"
    )
    collection.upsert(documents=documents, metadatas=metadata, ids=ids)
    logger.debug(
        f"Upserted {len(documents)} documents into collection '{
            collection_name}'"
    )


def log_data_fill_options(cleanup: bool, clean_content: bool, enable_wisdom: bool, fabric_command: str) -> None:
    """Log data fill options for user feedback."""
    if cleanup:
        logger.info("Cleanup enabled: collection will be deleted before filling.")
    if clean_content:
        logger.info(
            "Document cleaning enabled: HTML tags and UI elements will be removed before Markdown conversion."
        )
    else:
        logger.info(
            "Document cleaning disabled: raw HTML will be converted to Markdown without pre-cleaning."
        )
    if enable_wisdom:
        logger.info(
            f"Wisdom extraction enabled: {fabric_command} will be used to extract key insights."
        )


def process_data_fill(
    client: ClientAPI,
    collection_name: str,
    source_paths: list[str],
    mode: str,
    cleanup: bool,
    embedding_model: str,
    embedding_llm: str,
    embedding_ollama_host: str,
    embedding_ollama_port: int,
    bucket_name: str,
    bucket_path: str,
    clean_content: bool,
    enable_wisdom: bool,
    fabric_command: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Process data fill operation for multiple sources"""
    log_data_fill_options(cleanup, clean_content, enable_wisdom, fabric_command)
    logger.debug(
        f"Filling collection '{collection_name}' with data from {source_paths}"
    )

    if cleanup:
        delete_collection(client, collection_name)

    for source_path in source_paths:
        id_prefix = hashlib.sha256(source_path.encode()).hexdigest()[:20]
        logger.debug(f"Processing {source_path} with id prefix {id_prefix}")

        documents = load_documents(
            source_path=source_path,
            mode=mode,
            clean_content=clean_content,
            enable_wisdom=enable_wisdom,
            fabric_command=fabric_command,
        )

        if len(documents) == 0:
            logger.warning(f"No documents found in {source_path}. Skipping...")
            continue

        logger.debug(
            f"Bootstrapping collection '{collection_name}' with {
                len(documents)} documents"
        )
        bootstrap_db(
            client,
            collection_name,
            documents,
            embedding_model,
            embedding_llm,
            embedding_ollama_host,
            embedding_ollama_port,
            mode,
            id_prefix,
            bucket_name,
            bucket_path,
            chunk_size,
            chunk_overlap,
        )

    logger.debug(
        f"Collection '{
            collection_name}' has been created and filled with data."
    )


def search(
    client: ClientAPI, collection_name: str, query: str, embedding_function
) -> QueryResult:
    """Search for documents in the collection"""
    logger.debug(
        f"Searching collection '{
                 collection_name}' with query '{query}'"
    )

    """Search for documents in the collection and generate response"""
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )

    # https://docs.trychroma.com/docs/overview/getting-started
    results = collection.query(
        query_texts=[query],  # Chroma will embed this for you
        n_results=4,  # how many results to return
    )

    return results
