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
from .utils import format_content, sanitize_filename, parse_source_with_title, add_keybert_tags_to_doc
from .s3 import upload_to_s3
from .wisdom import extract_wisdom, check_fabric_installed

logger = logging.getLogger("RAG")


def process_data_fill(
    client: Optional[ClientAPI],
    args: argparse.Namespace,
) -> None:
    collection = None
    if client is not None and args.cleanup:
        delete_collection(client, args.collection)
        collection = create_get_collection(args, client, args.collection)

    # Set up OpenWebUIUploader if needed
    openwebui_uploader = None
    if args.upload_to_open_webui:
        openwebui_uploader = OpenWebUIUploader(args=args)

    for source_path_arg in args.source_path:
        source_path, override_title = parse_source_with_title(source_path_arg)
        documents = load_documents(source_path=source_path, args=args, override_title=override_title)
        if len(documents) == 0:
            logger.warning(f"No documents found in {source_path}. Skipping...")
            continue
        process_source_path(
            source_path=source_path,
            collection=collection,
            args=args,
            documents=documents,
            openwebui_uploader=openwebui_uploader,
            override_title=override_title,
        )


def process_source_path(
        source_path: str,
        collection: Collection | None,
        args: argparse.Namespace,
        documents: List[Document],
        openwebui_uploader: OpenWebUIUploader | None,
        override_title: str = ""
) -> None:
    fabric_installed = False

    if args.extract_wisdom:
        fabric_installed = check_fabric_installed(args.fabric_command)
        logger.info("Fabric command found, wisdom extraction possible")

    for i, doc in enumerate(documents):
        if override_title:
            doc.metadata["title"] = override_title
        doc.metadata["sanitized_title"] = sanitize_filename(doc.metadata["title"])  # Sanitized version for filenames

        wisdom = ""
        if args.extract_wisdom and fabric_installed:
            logger.debug("Extracting wisdom with Fabric")
            wisdom = extract_wisdom(doc.page_content, args.fabric_command, args.fabric_pattern)
            logger.info("Wisdom extracted successfully")

        doc = add_keybert_tags_to_doc(doc)

        doc = format_content(doc, wisdom)

        if openwebui_uploader:

            logger.info("Uploading documents to Open WebUI")

            # Add YAML frontmatter and format content
            content_with_metadata = "---\n"
            content_with_metadata += f"title: {doc.metadata.get('title', '')}\n"
            content_with_metadata += f"source: {source_path}\n"
            content_with_metadata += "---\n\n"
            content_with_metadata += doc.page_content

            # Debug logging before encoding
            logger.debug(f"Preparing to upload file: {doc.metadata['title']}")
            logger.debug("Content preview before encoding:")
            logger.debug(content_with_metadata[:500] + "..." if len(content_with_metadata) > 500 else content_with_metadata)

            file_content = content_with_metadata.encode("utf-8")
            content_size = len(file_content)
            logger.debug(f"Content size after encoding: {content_size} bytes")
            logger.debug(f"Metadata: {doc.metadata}")

            file_obj = io.BytesIO(file_content)
            file_obj.name = doc.metadata["sanitized_title"]  # requests uses this for the filename

            try:
                file_id = openwebui_uploader.upload_and_add(
                    file_obj,
                    filename=doc.metadata["title"]
                )
                if file_id:
                    logger.info(f'Uploaded {doc.metadata["title"]} to Open WebUI (file_id={file_id})')
                else:
                    logger.warning(f'Failed to upload {doc.metadata["title"]} to Open WebUI')
            except Exception as e:
                logger.error(f'Failed to upload {doc.metadata["title"]} to Open WebUI: {e}')

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
                else f'{args.bucket_path}/{doc.metadata["sanitized_title"]}.md'
            )

            doc.metadata["file_path"] = file_path
            upload_to_s3(doc.page_content, doc.metadata["sanitized_title"], args.bucket_path, args.bucket_name)

            logger.debug("S3 upload completed")

        if collection is not None:
            logger.debug(
                f"Filling collection '{args.collection}' with data from {args.source_paths}"
            )

            id_prefix = hashlib.sha256(source_path.encode()).hexdigest()[:20]
            logger.debug(f"Processing {source_path} with id prefix {id_prefix}")

            logger.debug(
                f"Inserting {len(documents)} documents into ChromaDB collection '{args.collection}'"
            )

            insert_into_collection(
                collection=collection,
                raw_documents=documents,
                args=args,
            )

            logger.debug(f"Collection '{args.collection}' has been created and filled with data.")
