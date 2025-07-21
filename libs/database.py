import os
import logging
import hashlib
from langchain_community.document_loaders import UnstructuredMarkdownLoader, AsyncHtmlLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .embedding import set_embedding_function

logger = logging.getLogger("RAG")


def delete_collection(client, collection):
    """Delete a collection from ChromaDB"""
    logger.info(f"Deleting collection '{collection}'")
    try:
        client.delete_collection(name=collection)
        logger.info(f"Collection '{collection}' deleted")
    except Exception as e:
        print(f"Error deleting collection {collection}: {e}")


def load_documents(source_path, source_type, mode):
    """Load documents from file or URL"""
    if source_type == "file":
        full_path = os.path.abspath(source_path)
        logger.info(f"Loading {full_path} with mode {mode}")
        loader = UnstructuredMarkdownLoader(full_path, mode=mode)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from {full_path}")
    else:
        # https://python.langchain.com/docs/how_to/document_loader_markdown/
        logger.info(f"Loading {source_path} with mode {mode}")
        urls = [source_path]
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents from {source_path}")

        # https://python.langchain.com/docs/integrations/document_transformers/markdownify/
        md = MarkdownifyTransformer()
        documents = md.transform_documents(docs)
        logger.info(f"Transformed {len(documents)} documents from {source_path}")

    return documents


def process_markdown_documents(chunks, mode, id_prefix):
    """Process markdown documents and extract metadata"""
    documents, metadata, ids = [], [], []
    logger.info(f"Processing {len(chunks)} chunks in {mode} mode")

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


def bootstrap_db(client, collection, raw_documents, llm, embedding_model, embedding_llm, mode, id_prefix):
    """Bootstrap the database with documents"""
    logger.info(f"Bootstrapping collection '{collection}' with {len(raw_documents)} documents")
    logger.info(f"Using Ollama embedding model '{embedding_model}'")
    embedding_function = set_embedding_function(embedding_llm, embedding_model)

    try:
        logger.info(f"Creating/getting collection {collection} with Ollama embedding function...")
        collection = client.get_or_create_collection(
            name=collection,
            embedding_function=embedding_function
        )
        logger.info(f"Collection '{collection}' created/gotten")
    except Exception as e:
        logger.error(f"Error creating/getting collection {collection}: {e}")

    logger.info(f"Splitting {len(raw_documents)} documents into chunks")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(raw_documents)
    logger.info(f"Split {len(raw_documents)} documents into {len(chunks)} chunks")

    documents, metadata, ids = process_markdown_documents(chunks, mode, id_prefix)

    logger.info(f"Upserting {len(documents)} documents into collection '{collection}'")
    collection.upsert(
        documents=documents,
        metadatas=metadata,
        ids=ids
    )
    logger.info(f"Upserted {len(documents)} documents into collection '{collection}'")


def process_data_fill(client, collection, source_paths, source_type, mode, cleanup, llm, embedding_model, embedding_llm):
    """Process data fill operation for multiple sources"""
    logger.info(f"Filling collection '{collection}' with data from {source_paths}")

    if cleanup:
        delete_collection(client, collection)

    for source_path in source_paths:
        id_prefix = hashlib.sha256(source_path.encode()).hexdigest()[:20]
        logger.info(f"Processing {source_path} with id prefix {id_prefix}")

        documents = load_documents(source_path, source_type, mode)

        logger.info(f"Bootstrapping collection '{collection}' with {len(documents)} documents")
        bootstrap_db(client, collection, documents, llm, embedding_model, embedding_llm, mode, id_prefix)

    logger.info(f"Collection '{collection}' has been created and filled with data.")
