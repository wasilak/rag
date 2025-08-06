import logging
from langchain_core.documents import Document
from typing import List, Optional
from chromadb.api.types import (
    Metadata,
    OneOrMany,
)
import hashlib

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
