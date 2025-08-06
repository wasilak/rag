import argparse
import logging
from .validation import validate_url
from langchain_core.documents import Document
from typing import List

from .documents_types.file import load_file_documents
from .documents_types.url import load_url_documents
from .utils import process_html_documents, convert_to_markdown

logger = logging.getLogger("RAG")


def load_documents(
    source_path: str,
    args: argparse.Namespace,
    override_title: str = None,
) -> List[Document]:
    # Use our validation module to check if path is a URL
    is_url = validate_url(source_path)
    # If file-based and --convert-to-markdown is set, treat as HTML and convert
    if not is_url and getattr(args, 'convert_to_markdown', False):
        docs = load_file_documents(source_path, args.mode, should_convert_to_markdown=True)
        docs = process_html_documents(docs, clean_content=args.clean_content)
        docs = convert_to_markdown(docs)
    elif is_url:
        docs = load_url_documents(source_path, args.clean_content)
    else:
        docs = load_file_documents(source_path, args.mode, should_convert_to_markdown=False)
    if override_title:
        for doc in docs:
            doc.metadata["title"] = override_title
    return docs
