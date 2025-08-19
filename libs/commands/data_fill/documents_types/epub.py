import argparse
import logging
import ebooklib
from ebooklib import epub
from typing import List
from langchain_core.documents import Document

from ..utils import get_title_from_file_name

logger = logging.getLogger("RAG")


def prepare_epub_documents(file_path: str, args: argparse.Namespace, override_title: str = None) -> List[Document]:
    """
    Parse an EPUB file and extract its content and metadata.

    Args:
        file_path (str): The path to the EPUB file.
        args (argparse.Namespace): The args namespace containing the convert_to_markdown flag.
        override_title (str, optional): The title to use instead of the one in the EPUB file. Defaults to None.

    Returns:
        List[Document]: A list of Document objects containing the extracted content and metadata.
    """
    try:
        book = epub.read_epub(file_path)
    except Exception as e:
        logger.error(f"Failed to read EPUB file {file_path}: {e}")
        return []

    epub_metadata = {}
    for key in book.metadata:
        for item in book.metadata[key]:
            epub_metadata[item] = book.metadata[key][item][0][0]

    docs_raw = []
    for document in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        try:
            content = document.content.decode("utf-8")
            docs_raw.append(content)
        except Exception as e:
            logger.warning(f"Failed to decode document content: {e}")
            continue

    # Check if there is any content
    if not docs_raw:
        logger.error(f"No content found in EPUB file {file_path}")
        return []

    # Combine all content and clean CSS code blocks
    combined_content = "\n\n".join(docs_raw)

    # Check if there is any content
    if not combined_content:
        logger.error(f"No content found in EPUB file {file_path}")
        return []

    docs = [Document(page_content=combined_content, metadata=epub_metadata)]

    # Check if docs is empty
    if not docs:
        logger.error(f"No documents found in EPUB file {file_path}")
        return []

    # Set original file_path as source for all loaded docs and merge with epub metadata
    for doc in docs:
        doc.metadata.update(epub_metadata)
        doc.metadata["source"] = file_path

        if override_title:
            doc.metadata["title"] = override_title

        # Add title to metadata if it's missing
        if "title" not in doc.metadata:
            doc.metadata["title"] = get_title_from_file_name(file_path)

    logger.debug(f"Loaded {len(docs)} documents from EPUB file {file_path}")
    return docs
