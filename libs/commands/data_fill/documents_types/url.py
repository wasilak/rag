import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import AsyncHtmlLoader
from ..utils import extract_title_from_html, process_html_documents, convert_to_markdown

logger = logging.getLogger("RAG")


def load_url_documents(
    url: str,
    clean_content: bool = False,
) -> List[Document]:
    logger.debug(f"Loading {url}")
    loader = AsyncHtmlLoader(
        [url],
        verify_ssl=False,
        header_template={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    docs = loader.load()
    logger.debug(f"Loaded {len(docs)} documents from {url}")

    # Extract and store titles before cleaning
    for doc in docs:
        title = doc.metadata.get("title", extract_title_from_html(doc.page_content))
        doc.metadata["title"] = title
        doc.metadata["source"] = url

    # Process documents
    docs = process_html_documents(docs, clean_content)
    docs = convert_to_markdown(docs)

    return docs
