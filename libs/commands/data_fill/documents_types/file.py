import logging
import os
from langchain_core.documents import Document
from typing import List
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from ..validation import validate_path, validate_file, validate_directory, validate_is_epub
from ..utils import convert_to_markdown as convert_to_markdown_func, medium_extract, process_html_documents, extract_title_from_html, apply_trafilatura, clean_html_content, remove_css_code_blocks
from langchain_core.documents import Document as LangDocument
import ebooklib
from ebooklib import epub

logger = logging.getLogger("RAG")


def load_file_documents(source_path: str, mode: str, should_convert_to_markdown: bool = False) -> List[Document]:
    try:
        full_path = os.path.abspath(source_path)
    except Exception as e:
        logger.error(f"Failed to resolve absolute path for {source_path}: {e}")
        return []

    if not validate_path(full_path):
        logger.error(f"Path does not exist: {full_path}")
        return []

    if validate_file(full_path):
        return load_file_document(full_path, mode, should_convert_to_markdown=should_convert_to_markdown)
    elif validate_directory(full_path):
        return load_directory_documents(full_path, mode, should_convert_to_markdown=should_convert_to_markdown)
    else:
        logger.error(f"Path {full_path} is not a valid file or directory")
        return []


def load_file_document(file_path: str, mode: str, should_convert_to_markdown: bool = False) -> List[Document]:
    try:
        is_epub = validate_is_epub(file_path)
    except Exception as e:
        logger.error(f"Failed to validate file type for {file_path}: {e}")
        return []

    logger.debug(f"Loading file {file_path} with mode {mode} (should_convert_to_markdown={should_convert_to_markdown})")

    if is_epub:
        return prepare_epub_documents(file_path)
    else:
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
            docs = [LangDocument(page_content=raw_content, metadata={})]
            docs = process_html_documents(docs, clean_content=True)
            docs = convert_to_markdown_func(docs)

            # Create Document objects directly instead of using temporary files
            documents = []
            for doc in docs:
                # Create Document object directly
                langchain_doc = LangDocument(page_content=doc.page_content, metadata={})
                documents.append(langchain_doc)

            # Set original file_path as source for all loaded docs
            for doc in documents:
                doc.metadata["source"] = file_path
        else:
            loader = UnstructuredMarkdownLoader(file_path, mode=mode)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = file_path

    for doc in documents:
        # Extract and store title from metadata
        try:
            title = doc.metadata.get("title", extract_title_from_html(doc.page_content))
        except Exception as e:
            logger.warning(f"Failed to extract title from document: {e}")
            title = "untitled"
        doc.metadata["title"] = title
    logger.debug(f"Loaded {len(documents)} documents from {file_path}")
    return documents


def load_directory_documents(directory: str, mode: str, should_convert_to_markdown: bool = False) -> List[Document]:
    logger.debug(f"Processing directory {directory} recursively")
    documents = []
    try:
        for root, _, files in os.walk(directory):
            markdown_files = [f for f in files if f.lower().endswith((".md", ".markdown"))]
            for file in markdown_files:
                file_path = os.path.join(root, file)
                docs = load_file_document(file_path, mode, should_convert_to_markdown=should_convert_to_markdown)
                documents.extend(docs)
    except Exception as e:
        logger.error(f"Failed to process directory {directory}: {e}")
        return []
    logger.debug(
        f"Loaded total of {len(documents)} documents from directory {directory}"
    )
    return documents


def prepare_epub_documents(file_path: str) -> List[Document]:
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

    # Combine all content and clean CSS code blocks
    combined_content = "\n\n".join(docs_raw)
    # Apply gentle cleaning focused on CSS code blocks
    combined_content = medium_extract(combined_content)
    combined_content = clean_html_content(combined_content)
    # Apply CSS code block removal without aggressive content extraction
    docs = [LangDocument(page_content=combined_content, metadata=epub_metadata)]
    # Apply gentle cleaning with CSS code block removal
    cleaned_content = medium_extract(combined_content)
    cleaned_content = clean_html_content(cleaned_content)
    # Apply CSS code block removal directly since clean_content=False doesn't do it
    cleaned_content = remove_css_code_blocks(cleaned_content)
    docs = [LangDocument(page_content=cleaned_content, metadata=epub_metadata)]
    docs = convert_to_markdown_func(docs)

    # Create Document objects directly instead of using temporary files
    documents = []
    for doc in docs:
        # Create Document object directly
        langchain_doc = LangDocument(page_content=doc.page_content, metadata={})
        documents.append(langchain_doc)

    # Set original file_path as source for all loaded docs and merge with epub metadata
    for doc in documents:
        doc.metadata.update(epub_metadata)
        doc.metadata["source"] = file_path
        # Extract and store title from metadata
        title = doc.metadata.get("title", extract_title_from_html(doc.page_content))
        doc.metadata["title"] = title

    logger.debug(f"Loaded {len(documents)} documents from EPUB file {file_path}")
    return documents
