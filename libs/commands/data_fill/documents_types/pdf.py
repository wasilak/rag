import logging
import argparse
from langchain_core.documents import Document
from pypdf import PdfReader
from ..utils import get_title_from_file_name
from typing import List


logger = logging.getLogger("RAG")


def prepare_pdf_documents(file_path: str, args: argparse.Namespace, override_title: str = "") -> List[Document]:
    try:
        reader = PdfReader(file_path)
        if reader is None:
            raise ValueError(f"Failed to read PDF file {file_path}")

        raw_text = ""

        logger.debug(f"Processing PDF file: {file_path} with {len(reader.pages)} pages")

        for page in reader.pages:
            if page is None:
                raise ValueError(f"Failed to read PDF page {page}")

            raw_text += page.extract_text()
            # Remove any layout or formatting characters
            raw_text = raw_text.replace("\n", " ").replace("\t", " ")

        if not raw_text:
            logger.warning("No text extracted from PDF file")
            return []

        # Create a single Document object with the raw text
        doc = Document(page_content=raw_text, metadata={})

        pdf_metadata = {}
        # Extract metadata from PDF file
        if reader.metadata:
            pdf_metadata = {key[1:]: value for key, value in reader.metadata.items() if key.startswith('/')}

        # normalize values as strings
        pdf_metadata = {key: str(value) for key, value in pdf_metadata.items()}

        doc.metadata.update(pdf_metadata)
        doc.metadata["source"] = file_path

        if override_title:
            doc.metadata["title"] = override_title

        # Add title to metadata if it's missing
        if "title" not in doc.metadata:
            doc.metadata["title"] = get_title_from_file_name(file_path)

        return [doc]
    except Exception as e:
        logger.error(f"Failed to read PDF file {file_path}", e)
        return []
