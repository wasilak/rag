import logging
import os
import tempfile
from langchain_core.documents import Document
from typing import List
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from ..validation import validate_path, validate_file, validate_directory, validate_is_epub
from ..utils import convert_to_markdown as convert_to_markdown_func, medium_extract, process_html_documents, extract_title_from_html, apply_trafilatura, clean_html_content
from langchain_core.documents import Document as LangDocument
import ebooklib
from ebooklib import epub

logger = logging.getLogger("RAG")


def load_file_documents(source_path: str, mode: str, should_convert_to_markdown: bool = False) -> List[Document]:
    full_path = os.path.abspath(source_path)

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
    is_epub = validate_is_epub(file_path)

    logger.debug(f"Loading file {file_path} with mode {mode} (should_convert_to_markdown={should_convert_to_markdown})")

    if is_epub:
        return prepare_epub_documents(file_path)
    else:
        if should_convert_to_markdown:
            # Read file as text (assume HTML or similar)
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            # Medium extraction step
            raw_content = medium_extract(raw_content)
            raw_content = clean_html_content(raw_content)
            raw_content = apply_trafilatura(raw_content)

            # Clean and convert to markdown using existing utils
            docs = [LangDocument(page_content=raw_content, metadata={})]
            docs = process_html_documents(docs, clean_content=True)
            docs = convert_to_markdown_func(docs)

            # Save to a temporary markdown file and load as markdown
            temp_md = tempfile.NamedTemporaryFile(delete=False, suffix='.md', mode='w', encoding='utf-8')
            temp_md.write(docs[0].page_content)
            temp_md.close()
            loader = UnstructuredMarkdownLoader(temp_md.name, mode=mode)
            documents = loader.load()
            os.unlink(temp_md.name)
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
        title = doc.metadata.get("title", extract_title_from_html(doc.page_content))
        doc.metadata["title"] = title
    logger.debug(f"Loaded {len(documents)} documents from {file_path}")
    return documents


def load_directory_documents(directory: str, mode: str, should_convert_to_markdown: bool = False) -> List[Document]:
    logger.debug(f"Processing directory {directory} recursively")
    documents = []
    for root, _, files in os.walk(directory):
        markdown_files = [f for f in files if f.lower().endswith((".md", ".markdown"))]
        for file in markdown_files:
            file_path = os.path.join(root, file)
            docs = load_file_document(file_path, mode, should_convert_to_markdown=should_convert_to_markdown)
            documents.extend(docs)
    logger.debug(
        f"Loaded total of {len(documents)} documents from directory {directory}"
    )
    return documents


def prepare_epub_documents(file_path: str) -> List[Document]:
    book = epub.read_epub(file_path)

    epub_metadata = {}
    for key in book.metadata:
        for item in book.metadata[key]:
            epub_metadata[item] = book.metadata[key][item][0][0]

    docs_raw = []
    for document in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        # docs.append(LangDocument(page_content=document.content.decode("utf-8"), metadata=epub_metadata))
        docs_raw.append(document.content.decode("utf-8"))

    docs = convert_to_markdown_func([LangDocument(page_content="\n\n".join(docs_raw), metadata=epub_metadata)])

    # Save to a temporary markdown file and load as markdown
    documents = []
    for doc in docs:
        temp_md = tempfile.NamedTemporaryFile(delete=False, suffix='.md', mode='w', encoding='utf-8')
        temp_md.write(doc.page_content)
        temp_md.close()

        loader = UnstructuredMarkdownLoader(temp_md.name, mode="single")
        documents.extend(loader.load())

        os.unlink(temp_md.name)

    # Set original file_path as source for all loaded docs
    for doc in documents:
        #  merge with epub metadata
        doc.metadata.update(epub_metadata)
        doc.metadata["source"] = file_path

    return documents
