import logging
import hashlib
import argparse
import io

from typing import List, Optional

from langchain_core.documents import Document

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from .openwebui import OpenWebUIUploader
from .validation import validate_s3_bucket_name, validate_s3_bucket_path
from .documents import load_documents
from .collection import delete_collection, create_get_collection, insert_into_collection
from .utils import log_data_fill_options, format_content, sanitize_filename
from .s3 import upload_markdown_to_s3
from .wisdom import extract_wisdom, check_fabric_installed

logger = logging.getLogger("RAG")


def process_data_fill(
    client: Optional[ClientAPI],
    args: argparse.Namespace,
) -> None:
    log_data_fill_options(
        cleanup=args.cleanup,
        clean_content=args.clean_content,
        enable_wisdom=args.extract_wisdom,
        fabric_command=args.fabric_command,
    )

    collection = None
    if client is not None and args.cleanup:
        delete_collection(client, args.collection)
        collection = create_get_collection(client, args.collection)

    # Set up OpenWebUIUploader if needed
    openwebui_uploader = None
    if args.upload_to_open_webui:
        openwebui_uploader = OpenWebUIUploader(
            api_url=args.open_webui_url,
            api_key=args.open_webui_api_key,
            knowledge_id=args.open_webui_knowledge_id or None,
        )

    for source_path in args.source_path:

        documents = load_documents(
            source_path=source_path,
            mode=args.mode,
            clean_content=args.clean_content,
            enable_wisdom=args.extract_wisdom,
            fabric_command=args.fabric_command,
        )

        if len(documents) == 0:
            logger.warning(f"No documents found in {source_path}. Skipping...")
            continue

        process_source_path(
            source_path=source_path,
            collection_name=args.collection,
            collection=collection,
            args=args,
            documents=documents,
            openwebui_uploader=openwebui_uploader,
        )


def process_source_path(
        source_path: str,
        collection_name: str,
        collection: Collection,
        args: argparse.Namespace,
        documents: List[Document],
        openwebui_uploader: OpenWebUIUploader
) -> None:

    if args.extract_wisdom:
        fabric_installed = check_fabric_installed(args.fabric_command)
        logger.info("Fabric command found, wisdom extraction possible")

    for i, doc in enumerate(documents):

        doc.metadata["sanitized_title"] = sanitize_filename(doc.metadata["title"])  # Sanitized version for filenames

        wisdom = ""
        if args.extract_wisdom and fabric_installed:
            logger.debug("Extracting wisdom with Fabric")
            wisdom = extract_wisdom(doc.page_content, args.fabric_command)
            logger.info("Wisdom extracted successfully")

        doc = format_content(doc, wisdom)

        if openwebui_uploader:

            logger.info("Uploading documents to Open WebUI")

            file_content = doc.page_content.encode("utf-8")
            file_obj = io.BytesIO(file_content)
            file_obj.name = doc.metadata["sanitized_title"]  # requests uses this for the filename

            try:
                file_id = openwebui_uploader.upload_and_add(file_obj, filename=doc.metadata["title"])
                if file_id:
                    logger.info(f"Uploaded {doc.metadata["title"]} to Open WebUI (file_id={file_id})")
                else:
                    logger.warning(f"Failed to upload {doc.metadata["title"]} to Open WebUI")
            except Exception as e:
                logger.error(f"Failed to upload {doc.metadata["title"]} to Open WebUI: {e}")

        # S3 upload logic (if enabled and Chroma is skipped)
        if args.upload_to_s3:
            if not args.bucket_name:
                logger.error("S3 upload requested but no bucket name provided.")
            elif not validate_s3_bucket_name(args.bucket_name):
                logger.error(f"Invalid S3 bucket name: {args.bucket_name}")
            elif not validate_s3_bucket_path(args.bucket_path):
                logger.error(f"Invalid S3 bucket path: {args.bucket_path}")
            else:
                logger.info("Uploading documents to S3")

            # Add bucket_path to metadata
            doc.metadata["bucket_path"] = args.bucket_path

            file_path = (
                doc.metadata["sanitized_title"] + ".md"
                if not args.bucket_path
                else f"{args.bucket_path}/{doc.metadata["sanitized_title"]}.md"
            )

            doc.metadata["file_path"] = file_path
            upload_markdown_to_s3(doc.page_content, doc.metadata["sanitized_title"], args.bucket_path, args.bucket_name)

            logger.debug("S3 upload completed")

        if collection is not None:
            logger.debug(
                f"Filling collection '{args.collection}' with data from {args.source_paths}"
            )

            id_prefix = hashlib.sha256(source_path.encode()).hexdigest()[:20]
            logger.debug(f"Processing {source_path} with id prefix {id_prefix}")

            logger.debug(
                f"Inserting {len(documents)} documents into ChromaDB collection '{collection_name}'"
            )

            insert_into_collection(
                collection=collection,
                documents=documents,
                args=args,
            )

            logger.debug(f"Collection '{args.collection}' has been created and filled with data.")
