import logging
import os
from langchain_core.documents import Document
from typing import List
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from ..validation import validate_path, validate_file, validate_directory
from ..utils import extract_title_from_html

logger = logging.getLogger("RAG")


def load_file_documents(source_path: str, mode: str) -> List[Document]:
    full_path = os.path.abspath(source_path)

    if not validate_path(full_path):
        logger.error(f"Path does not exist: {full_path}")
        return []

    if validate_file(full_path):
        return load_file_document(full_path, mode)
    elif validate_directory(full_path):
        return load_directory_documents(full_path, mode)
    else:
        logger.error(f"Path {full_path} is not a valid file or directory")
        return []


def load_file_document(file_path: str, mode: str) -> List[Document]:
    logger.debug(f"Loading file {file_path} with mode {mode}")
    loader = UnstructuredMarkdownLoader(file_path, mode=mode)
    documents = loader.load()

    for doc in documents:
        # Extract and store title from metadata
        title = doc.metadata.get("title", extract_title_from_html(doc.page_content))
        doc.metadata["title"] = title
        doc.metadata["source"] = file_path

    logger.debug(f"Loaded {len(documents)} documents from {file_path}")
    return documents


def load_directory_documents(directory: str, mode: str) -> List[Document]:
    logger.debug(f"Processing directory {directory} recursively")
    documents = []
    for root, _, files in os.walk(directory):
        markdown_files = [f for f in files if f.lower().endswith((".md", ".markdown"))]
        for file in markdown_files:
            file_path = os.path.join(root, file)
            docs = load_file_document(file_path, mode)
            documents.extend(docs)
    logger.debug(
        f"Loaded total of {len(documents)} documents from directory {directory}"
    )
    return documents
