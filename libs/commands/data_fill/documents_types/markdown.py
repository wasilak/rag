import logging
import argparse
from langchain_core.documents import Document
from typing import List, Optional
import hashlib

from chromadb.api.types import (
    Metadata,
    OneOrMany,
)
from langchain_community.document_loaders import UnstructuredMarkdownLoader

from ..utils import convert_to_markdown, get_title_from_file_name, medium_extract, process_html_documents, extract_title_from_html, apply_trafilatura, clean_html_content

logger = logging.getLogger("RAG")


def process_markdown_documents(
    chunks: List[Document], mode: str, id_prefix: str
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
            ids.append(f"{id_prefix}_{i}")  # Consistent f-string usage
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
            # Use a hash of id_prefix and element_id for uniqueness
            unique_element_id = hashlib.sha256(f"{id_prefix}_{element_id}".encode()).hexdigest()[:20]
            ids.append(unique_element_id)

            temp_metadata = {}
            for key, value in chunk.metadata.items():
                temp_metadata[key] = (
                    ", ".join(value) if isinstance(value, list) else value
                )

            if parent_chain_docs:
                temp_metadata["page_title"] = parent_chain_docs[0].page_content
                temp_metadata["sanitized_title"] = (
                    chunk.page_content
                    if chunk.metadata.get("category") == "Title"
                    else parent_chain_docs[-1].page_content
                )
            else:
                temp_metadata["page_title"] = chunk.page_content
                temp_metadata["sanitized_title"] = chunk.page_content

            metadata.append(temp_metadata)

    return documents, metadata, ids


def prepare_markdown_documents(
    file_path: str,
    args: argparse.Namespace,
    should_convert_to_markdown: bool,
    override_title: str | None
) -> List[Document]:

    if should_convert_to_markdown:
        try:
            # Read file as text (assume HTML or similar)
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

        # Medium extraction step
        raw_content = medium_extract(raw_content)
        raw_content = clean_html_content(raw_content)
        raw_content = apply_trafilatura(raw_content)

        # Clean and convert to markdown using existing utils
        docs = [Document(page_content=raw_content, metadata={})]
        docs = process_html_documents(docs, clean_content=True)
        docs = convert_to_markdown(docs)

        # Create Document objects directly instead of using temporary files
        documents = []
        for doc in docs:
            # Create Document object directly
            langchain_doc = Document(page_content=doc.page_content, metadata={})
            documents.append(langchain_doc)

        # Set original file_path as source for all loaded docs
        for doc in documents:
            doc.metadata["source"] = file_path
    else:
        loader = UnstructuredMarkdownLoader(file_path, mode=args.mode)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = file_path

    for doc in documents:
        # Extract and store title from metadata
        if override_title:
            doc.metadata["title"] = override_title

        # Add title to metadata if it's missing
        if "title" not in doc.metadata:
            doc.metadata["title"] = get_title_from_file_name(file_path)

        doc.metadata["title"] = doc.metadata.get("title", extract_title_from_html(doc.page_content))

    return documents
