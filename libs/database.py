import os
import logging
import hashlib
from langchain_community.document_loaders import UnstructuredMarkdownLoader, AsyncHtmlLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from libs.s3 import upload_markdown_to_s3
from .embedding import set_embedding_function
from chromadb.api import ClientAPI
from typing import Sequence, List, Optional
from langchain_core.documents import Document
from chromadb.api.types import (
    Metadata,
    QueryResult,
    OneOrMany,
)
from .utils import sanitize_title
from bs4 import BeautifulSoup

logger = logging.getLogger("RAG")


def delete_collection(client: ClientAPI, collection: str) -> None:
    """Delete a collection from ChromaDB"""
    logger.debug(f"Deleting collection '{collection}'")
    try:
        client.delete_collection(name=collection)
        logger.debug(f"Collection '{collection}' deleted")
    except Exception as e:
        print(f"Error deleting collection {collection}: {e}")


def clean_html(raw_html: str) -> str:
    """Clean HTML content by removing unwanted elements and keeping main content"""
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


def load_documents(source_path: str, source_type: str, mode: str, clean_content: bool = False) -> list[Document] | Sequence[Document]:
    """Load documents from file or URL"""
    if source_type == "file":
        full_path = os.path.abspath(source_path)

        if os.path.isfile(full_path):
            # Handle single file
            logger.debug(f"Loading file {full_path} with mode {mode}")
            loader = UnstructuredMarkdownLoader(full_path, mode=mode)
            documents = loader.load()
            logger.debug(f"Loaded {len(documents)} documents from {full_path}")
            return documents
        elif os.path.isdir(full_path):
            # Handle directory recursively
            logger.debug(f"Processing directory {full_path} recursively")
            documents = []
            for root, _, files in os.walk(full_path):
                markdown_files = [f for f in files if f.lower().endswith(('.md', '.markdown'))]
                for file in markdown_files:
                    file_path = os.path.join(root, file)
                    logger.debug(f"Loading file {file_path} with mode {mode}")
                    loader = UnstructuredMarkdownLoader(file_path, mode=mode)
                    docs = loader.load()
                    logger.debug(f"Loaded {len(docs)} documents from {file_path}")
                    documents.extend(docs)
            logger.debug(f"Loaded total of {len(documents)} documents from directory {full_path}")
            return documents
        else:
            logger.error(f"Source path {full_path} is not a valid file or directory")
            return []
    else:
        # https://python.langchain.com/docs/how_to/document_loader_markdown/
        logger.debug(f"Loading {source_path} with mode {mode}")
        urls = [source_path]
        # Configure loader with extended headers for better content access
        loader = AsyncHtmlLoader(
            urls,
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
        logger.debug(f"Loaded {len(docs)} documents from {source_path}")

        # Handle HTML cleaning if enabled
        if clean_content:
            # Calculate content reduction after cleaning
            total_original = sum(len(doc.page_content) for doc in docs)
            cleaned_docs = []
            for doc in docs:
                cleaned_html = clean_html(doc.page_content)
                doc.page_content = cleaned_html
                cleaned_docs.append(doc)
            total_cleaned = sum(len(doc.page_content) for doc in cleaned_docs)
            reduction_percent = ((total_original - total_cleaned) / total_original * 100) if total_original > 0 else 0
            logger.info(f"Document cleaning enabled: HTML tags and UI elements will be removed before Markdown conversion. Content reduced from {total_original} to {total_cleaned} chars ({reduction_percent:.1f}% reduction).")
        else:
            logger.info("Document cleaning disabled: raw HTML will be converted to Markdown without pre-cleaning.")
            cleaned_docs = docs

        # Documents are already cleaned if clean_content was enabled

        # Configure markdownify transformer
        # https://python.langchain.com/docs/integrations/document_transformers/markdownify/
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
        documents = md.transform_documents(cleaned_docs)
        logger.debug(f"Transformed {len(documents)} documents from {source_path}")
        for i, doc in enumerate(documents):
            logger.debug(f"Document {i+1} after Markdownify (first 500 chars):\n{doc.page_content[:500]}...")

    return documents


def process_markdown_documents(chunks: Sequence[Document], mode: str, id_prefix: str) -> tuple[List[str], Optional[OneOrMany[Metadata]], List[str]]:
    """Process markdown documents and extract metadata"""
    documents: List[str] = []
    metadata: Optional[OneOrMany[Metadata]] = []
    ids: List[str] = []
    logger.debug(f"Processing {len(chunks)} chunks in {mode} mode")

    if mode == "single":
        i = 0
        for chunk in chunks:
            documents.append(chunk.page_content)
            ids.append(id_prefix+"_"+str(i))
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
            ids.append(f"ID_{i}")

            temp_metadata = {}
            for key, value in chunk.metadata.items():
                temp_metadata[key] = ', '.join(value) if isinstance(value, list) else value

            if parent_chain_docs:
                temp_metadata["page_title"] = parent_chain_docs[0].page_content
                temp_metadata["resolved_title"] = chunk.page_content if chunk.metadata.get("category") == "Title" else parent_chain_docs[-1].page_content
            else:
                temp_metadata["page_title"] = chunk.page_content
                temp_metadata["resolved_title"] = chunk.page_content

            metadata.append(temp_metadata)

    return documents, metadata, ids


def bootstrap_db(client: ClientAPI, collection_name: str, raw_documents, embedding_model: str, embedding_llm: str, mode: str, id_prefix: str, source_type: str, bucket_name: str, bucket_path: str, enable_cleaning: bool = False, cleaning_llm: str = "ollama", cleaning_model: str = "qwen3:8b", cleaning_prompt: Optional[str] = None) -> None:
    """Bootstrap the database with documents"""
    logger.debug(f"Bootstrapping collection '{collection_name}' with {len(raw_documents)} documents")
    logger.debug(f"Using embedding model '{embedding_model}' with provider '{embedding_llm}'")

    # LLM cleanup removed: documents are now cleaned only via HTML/Markdown pre-processing

    # Upload to S3 if enabled (after cleaning, before chunking)
    if source_type == "url" and len(bucket_name) != 0:
        logger.info("Uploading documents to S3")
        for doc in raw_documents:
            sanitized_title = sanitize_title(doc.metadata.get("title", "untitled"))
            doc.metadata["file_path"] = f"{bucket_path}/{sanitized_title}.md"
            upload_markdown_to_s3(doc.page_content, sanitized_title, bucket_path, bucket_name)
        logger.debug("S3 upload completed")

    embedding_function = set_embedding_function(embedding_llm, embedding_model)

    try:
        logger.debug(f"Creating/getting collection {collection_name} with Ollama embedding function...")
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        logger.debug(f"Collection '{collection}' created/gotten")
    except Exception as e:
        logger.error(f"Error creating/getting collection {collection_name}: {e}")
        exit(1)

    logger.debug(f"Splitting {len(raw_documents)} documents into chunks")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(raw_documents)
    logger.debug(f"Split {len(raw_documents)} documents into {len(chunks)} chunks")

    documents, metadata, ids = process_markdown_documents(chunks, mode, id_prefix)

    logger.debug(f"Upserting {len(documents)} documents into collection '{collection_name}'")
    collection.upsert(
        documents=documents,
        metadatas=metadata,
        ids=ids
    )
    logger.debug(f"Upserted {len(documents)} documents into collection '{collection_name}'")


def process_data_fill(client: ClientAPI, collection_name: str, source_paths: list[str], source_type: str, mode: str, cleanup: bool, embedding_model: str, embedding_llm: str, bucket_name: str, bucket_path: str, clean_content: bool = False) -> None:
    """Process data fill operation for multiple sources"""
    logger.debug(f"Filling collection '{collection_name}' with data from {source_paths}")

    if cleanup:
        delete_collection(client, collection_name)

    for source_path in source_paths:
        id_prefix = hashlib.sha256(source_path.encode()).hexdigest()[:20]
        logger.debug(f"Processing {source_path} with id prefix {id_prefix}")

        documents = load_documents(source_path, source_type, mode, clean_content=clean_content)

        if len(documents) == 0:
            logger.warning(f"No documents found in {source_path}. Skipping...")
            continue

        logger.debug(f"Bootstrapping collection '{collection_name}' with {len(documents)} documents")
        bootstrap_db(client, collection_name, documents, embedding_model, embedding_llm, mode, id_prefix, source_type, bucket_name, bucket_path)

    logger.debug(f"Collection '{collection_name}' has been created and filled with data.")

def search(client: ClientAPI, collection_name: str, query: str, embedding_function) -> QueryResult:
    """Search for documents in the collection"""
    logger.debug(f"Searching collection '{collection_name}' with query '{query}'")

    """Search for documents in the collection and generate response"""
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

    # https://docs.trychroma.com/docs/overview/getting-started
    results = collection.query(
        query_texts=[query],  # Chroma will embed this for you
        n_results=4  # how many results to return
    )

    return results
