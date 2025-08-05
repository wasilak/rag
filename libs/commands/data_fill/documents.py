import logging
from .validation import validate_url
from langchain_core.documents import Document
from typing import List

from .documents_types.file import load_file_documents
from .documents_types.url import load_url_documents

logger = logging.getLogger("RAG")


def load_documents(
    source_path: str,
    mode: str,
    clean_content: bool = False,
    enable_wisdom: bool = False,
    fabric_command: str = "fabric",
) -> List[Document]:

    # Use our validation module to check if path is a URL
    is_url = validate_url(source_path)

    if is_url:
        docs = load_url_documents(source_path, clean_content)
    else:
        docs = load_file_documents(source_path, mode)

    return docs
