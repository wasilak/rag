"""Document loading and processing module."""
import os
import logging
from typing import List, Sequence
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader, AsyncHtmlLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from bs4 import BeautifulSoup

from .wisdom import extract_wisdom, format_content

logger = logging.getLogger("RAG")


def load_file_document(file_path: str, mode: str) -> List[Document]:
    """Load a single file document.

    Args:
        file_path: Path to the file
        mode: Processing mode ('single' or 'elements')

    Returns:
        List of loaded documents
    """
    logger.debug(f"Loading file {file_path} with mode {mode}")
    loader = UnstructuredMarkdownLoader(file_path, mode=mode)
    documents = loader.load()
    logger.debug(f"Loaded {len(documents)} documents from {file_path}")
    return documents


def load_directory_documents(directory: str, mode: str) -> List[Document]:
    """Load documents from a directory recursively.

    Args:
        directory: Directory path
        mode: Processing mode ('single' or 'elements')

    Returns:
        List of loaded documents
    """
    logger.debug(f"Processing directory {directory} recursively")
    documents = []
    for root, _, files in os.walk(directory):
        markdown_files = [f for f in files if f.lower().endswith(('.md', '.markdown'))]
        for file in markdown_files:
            file_path = os.path.join(root, file)
            docs = load_file_document(file_path, mode)
            documents.extend(docs)
    logger.debug(f"Loaded total of {len(documents)} documents from directory {directory}")
    return documents


def load_file_documents(source_path: str, mode: str) -> List[Document]:
    """Load documents from file or directory.

    Args:
        source_path: Path to file or directory
        mode: Processing mode ('single' or 'elements')

    Returns:
        List of loaded documents
    """
    full_path = os.path.abspath(source_path)

    if os.path.isfile(full_path):
        return load_file_document(full_path, mode)
    elif os.path.isdir(full_path):
        return load_directory_documents(full_path, mode)
    else:
        logger.error(f"Source path {full_path} is not a valid file or directory")
        return []


def clean_html_content(raw_html: str) -> str:
    """Clean HTML content by removing unwanted elements and keeping main content.

    Args:
        raw_html: Raw HTML content

    Returns:
        Cleaned HTML content
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    # Remove unwanted tags typical for online publications
    for tag in soup.find_all([
        'nav', 'header', 'footer', 'aside', 'script', 'style', 'noscript', 'form', 'dialog', 'menu', 'iframe',
        'template', 'banner', 'search', 'toolbar', 'menubar', 'contentinfo', 'complementary', 'advertisement',
        'comments', 'sidebar', 'cookie', 'popup', 'modal', 'social', 'share', 'newsletter', 'widget',
        'promotion', 'ad', 'subscribe', 'related', 'recommended'
    ]):
        tag.decompose()
    # Keep only main content tags if present
    main_content = None
    for main_tag in ['article', 'main']:
        main_content = soup.find(main_tag)
        if main_content:
            break
    if main_content:
        return str(main_content)
    return str(soup)


def process_html_documents(docs: List[Document], clean_content: bool = False) -> List[Document]:
    """Process HTML documents with optional cleaning.

    Args:
        docs: List of documents
        clean_content: Whether to clean HTML content

    Returns:
        Processed documents
    """
    if clean_content:
        total_original = sum(len(doc.page_content) for doc in docs)
        cleaned_docs = []
        for doc in docs:
            cleaned_html = clean_html_content(doc.page_content)
            doc.page_content = cleaned_html
            cleaned_docs.append(doc)
        total_cleaned = sum(len(doc.page_content) for doc in cleaned_docs)
        reduction_percent = ((total_original - total_cleaned) / total_original * 100) if total_original > 0 else 0
        logger.info(f"HTML tags and UI elements removed before Markdown conversion. Content reduced from {total_original} to {total_cleaned} chars ({reduction_percent:.1f}% reduction).")
    else:
        logger.info("Document cleaning disabled: raw HTML will be converted to Markdown without pre-cleaning.")
        cleaned_docs = docs

    return cleaned_docs


def convert_to_markdown(docs: List[Document]) -> Sequence[Document]:
    """Convert HTML documents to Markdown.

    Args:
        docs: List of documents

    Returns:
        Documents converted to Markdown
    """
    logger.debug("Configuring Markdownify transformer")
    md = MarkdownifyTransformer(
        strip=['script', 'style', 'meta', 'link', 'iframe', 'button', 'input', 'select', 'textarea'],
        remove=['nav', 'header', 'footer', 'aside', 'noscript', 'head',
               'template', 'dialog', 'menu', 'form', 'banner', 'search',
               'toolbar', 'menubar', 'contentinfo', 'complementary',
               'advertisement', 'comments', 'sidebar', 'cookie', 'popup',
               'modal', 'social', 'share', 'newsletter', 'widget',
               'promotion', 'ad', 'subscribe', 'related', 'recommended'],
        heading_style="ATX",
        bullets="-",
        wrap=0,
        preserve_images=True,
        emphasis_mark="*",
        strong_mark="**",
        escape_asterisks=False,
        code_language="",
        default_title=True,
        newline_style="\n",
        keep_formatting=True
    )
    return md.transform_documents(docs)


def extract_title_from_html(content: str) -> str:
    """Extract title from HTML content.

    Args:
        content: HTML content

    Returns:
        Extracted title or 'untitled'
    """
    soup = BeautifulSoup(content, "html.parser")
    title_tag = soup.find('title')
    if title_tag and title_tag.get_text(strip=True):
        return title_tag.get_text(strip=True)
    h1_tag = soup.find('h1')
    if h1_tag and h1_tag.get_text(strip=True):
        return h1_tag.get_text(strip=True)
    return "untitled"


def process_wisdom_extraction(docs: Sequence[Document], fabric_command: str) -> Sequence[Document]:
    """Process wisdom extraction for documents.

    Args:
        docs: List of documents
        fabric_command: Fabric command to use

    Returns:
        Processed documents, potentially split into wisdom/original pairs
    """
    from .wisdom import check_fabric_installed

    if not check_fabric_installed(fabric_command):
        logger.warning(f"Fabric command '{fabric_command}' not found. Continuing without wisdom extraction.")
        return docs

    logger.info("Fabric command found, proceeding with wisdom extraction")
    processed_docs = []

    for doc in docs:
        logger.debug("Extracting wisdom with Fabric")
        wisdom = extract_wisdom(doc.page_content, fabric_command)
        if wisdom:
            # Create two versions of the document
            base_title = doc.metadata.get("base_title")
            if not base_title:
                # Fall back to the title field if base_title is not set
                base_title = doc.metadata.get("title", "untitled")
                doc.metadata["base_title"] = base_title
            bucket_path = doc.metadata.get("bucket_path", "")
            wisdom_content, original_content = format_content(doc.page_content, base_title, wisdom, bucket_path)

            # Create wisdom document
            wisdom_doc = Document(
                page_content=wisdom_content,
                metadata={**doc.metadata, "is_wisdom": True}
            )
            processed_docs.append(wisdom_doc)

            # Create original document
            original_doc = Document(
                page_content=original_content,
                metadata={**doc.metadata, "is_original": True}
            )
            processed_docs.append(original_doc)
            logger.info("Created wisdom and original versions of the document")
        else:
            # If wisdom extraction fails, use original content
            doc.page_content = doc.page_content.strip()
            processed_docs.append(doc)
            logger.warning("Failed to extract wisdom, continuing with original content only")

    return processed_docs


def load_url_documents(url: str, clean_content: bool = False, enable_wisdom: bool = False, fabric_command: str = 'fabric') -> Sequence[Document]:
    """Load documents from URL.

    Args:
        url: URL to load
        clean_content: Whether to clean HTML content
        enable_wisdom: Whether to enable wisdom extraction
        fabric_command: Fabric command to use

    Returns:
        List of loaded documents
    """
    logger.debug(f"Loading {url}")
    loader = AsyncHtmlLoader(
        [url],
        verify_ssl=False,
        header_template={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }
    )
    docs = loader.load()
    logger.debug(f"Loaded {len(docs)} documents from {url}")

    # Extract and store titles before cleaning
    for doc in docs:
        title = doc.metadata.get("title", extract_title_from_html(doc.page_content))
        doc.metadata["title"] = title
        doc.metadata["base_title"] = title  # Set base_title as well

    # Process documents
    docs = process_html_documents(docs, clean_content)
    docs = convert_to_markdown(docs)

    # Extract wisdom if enabled
    if enable_wisdom:
        docs = process_wisdom_extraction(docs, fabric_command)

    # Log final documents
    for i, doc in enumerate(docs):
        logger.debug(f"Document {i+1} final content (first 500 chars):\n{doc.page_content[:500]}...")

    return docs


def load_documents(
      source_path: str,
      source_type: str,
      mode: str,
      clean_content: bool = False,
      enable_wisdom: bool = False,
      fabric_command: str = 'fabric'
    ) -> List[Document] | Sequence[Document]:
    """Load documents from file or URL.

    Args:
        source_path: Path to file/directory or URL
        source_type: Type of source ('file' or 'url')
        mode: Processing mode ('single' or 'elements')
        clean_content: Whether to clean HTML content
        enable_wisdom: Whether to enable wisdom extraction
        fabric_command: Fabric command to use

    Returns:
        List of loaded documents
    """
    if source_type == "file":
        return load_file_documents(source_path, mode)
    else:
        return load_url_documents(source_path, clean_content, enable_wisdom, fabric_command)
