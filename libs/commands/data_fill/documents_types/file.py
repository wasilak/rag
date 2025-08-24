import argparse
import logging
import os
from langchain_core.documents import Document
from typing import List

from ..validation import validate_is_pdf, validate_file, validate_directory, validate_is_epub
from .epub import prepare_epub_documents
from .pdf import prepare_pdf_documents
from .markdown import prepare_markdown_documents


logger = logging.getLogger("RAG")


def load_file_documents(source_path: str, args: argparse.Namespace, should_convert_to_markdown: bool = False, override_title: str = "") -> List[Document]:
    try:
        full_path = os.path.abspath(source_path)
    except Exception as e:
        logger.error(f"Failed to resolve absolute path for {source_path}: {e}")
        return []

    full_path = str(full_path).replace('\\', '')

    if validate_file(full_path):
        return load_file_document(full_path, args, should_convert_to_markdown=should_convert_to_markdown, override_title=override_title)
    elif validate_directory(full_path):
        return load_directory_documents(full_path, args, should_convert_to_markdown=should_convert_to_markdown, override_title=override_title)
    else:
        logger.error(f"Path {full_path} is not a valid file or directory")
        return []


def load_file_document(file_path: str, args: argparse.Namespace, should_convert_to_markdown: bool = False, override_title: str = "") -> List[Document]:
    try:
        is_epub = validate_is_epub(file_path)
    except Exception as e:
        logger.error(f"Failed to validate file type for {file_path}: {e}")
        return []

    try:
        is_pdf = validate_is_pdf(file_path)
    except Exception as e:
        logger.error(f"Failed to validate file type for {file_path}: {e}")
        return []

    logger.debug(f"Loading file {file_path} with mode {args.mode} (should_convert_to_markdown={should_convert_to_markdown})")

    if is_epub:
        documents = prepare_epub_documents(file_path, args=args, override_title=override_title)
    elif is_pdf:
        documents = prepare_pdf_documents(file_path, args=args, override_title=override_title)
    else:
        documents = prepare_markdown_documents(file_path, args=args, should_convert_to_markdown=should_convert_to_markdown, override_title=override_title)

    logger.debug(f"Loaded {len(documents)} documents from {file_path}")
    return documents


def load_directory_documents(directory: str, args: argparse.Namespace, should_convert_to_markdown: bool = False, override_title: str = "") -> List[Document]:
    logger.debug(f"Processing directory {directory} recursively")
    documents = []
    try:
        for root, _, files in os.walk(directory):
            markdown_files = [f for f in files if f.lower().endswith((".md", ".markdown"))]
            for file in markdown_files:
                file_path = os.path.join(root, file)
                docs = load_file_document(file_path, args, should_convert_to_markdown=should_convert_to_markdown, override_title=override_title)
                documents.extend(docs)
    except Exception as e:
        logger.error(f"Failed to process directory {directory}: {e}")
        return []
    logger.debug(
        f"Loaded total of {len(documents)} documents from directory {directory}"
    )
    return documents
