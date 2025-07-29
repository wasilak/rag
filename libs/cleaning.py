import logging
from typing import List, Optional
from openai import OpenAI
from langchain_core.documents import Document
from .models import get_best_model

logger = logging.getLogger("RAG")


class DocumentCleaner:
    """Clean documents using LLM to remove ads, navigation, and obsolete content"""

    def __init__(self, llm_provider: str, model: str, embedding_ollama_host: str, embedding_ollama_port: int):
        """Initialize document cleaner

        Args:
            llm_provider: LLM provider (ollama, openai, gemini)
            model: Model name to use for cleaning
        """
        self.llm_provider = llm_provider
        self.model = get_best_model(llm_provider, model, "chat")
        self.client = self._create_llm_client()
        self.embedding_ollama_host = embedding_ollama_host
        self.embedding_ollama_port = embedding_ollama_port

    def _create_llm_client(self) -> OpenAI:
        """Create LLM client based on the provider"""
        import os

        if self.llm_provider == "ollama":
            logger.debug("Using Ollama for document cleaning")
            return OpenAI(
                base_url=f'http://{self.embedding_ollama_host}:{self.embedding_ollama_port}/v1',
                api_key='ollama',  # required, but unused
            )
        elif self.llm_provider == "gemini":
            logger.debug("Using Gemini for document cleaning")
            return OpenAI(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:
            logger.debug("Using OpenAI for document cleaning")
            return OpenAI()

    def clean_document(self, document: Document, cleaning_prompt: Optional[str] = None) -> Document:
        """Clean a single document using LLM

        Args:
            document: Document to clean
            cleaning_prompt: Custom cleaning prompt (optional)

        Returns:
            Cleaned document
        """
        if not cleaning_prompt:
            cleaning_prompt = self._get_default_cleaning_prompt()

        logger.debug(f"Cleaning document with model {self.model}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": cleaning_prompt},
                    {"role": "user", "content": document.page_content}
                ],
                temperature=0.1  # Low temperature for consistent cleaning
            )

            if response.choices and response.choices[0].message.content:
                cleaned_content = response.choices[0].message.content.strip()
                # Remove thinking tags and their content
                import re
                cleaned_content = re.sub(r'<think>.*?</think>', '', cleaned_content, flags=re.DOTALL)
                cleaned_content = cleaned_content.strip()

                # Log the cleaned content for inspection
                logger.debug("=== Cleaned content start ===")
                logger.debug(cleaned_content[:2000])  # First 2000 chars
                logger.debug("=== Cleaned content end ===")

                # Log the original content for comparison
                logger.debug("=== Original content start ===")
                logger.debug(document.page_content[:2000])  # First 2000 chars
                logger.debug("=== Original content end ===")

                # Validate content length
                original_length = len(document.page_content)
                cleaned_length = len(cleaned_content)
                reduction_percent = ((original_length - cleaned_length) / original_length) * 100

                # If more than 30% of content is removed, likely a summary was created
                if reduction_percent > 30:
                    logger.warning(f"Excessive content reduction detected ({reduction_percent:.1f}%). Using original content.")
                    return document

                # Check for summary markers
                summary_markers = ['here\'s', 'key points', 'overview', 'summary', 'this article', 'this guide']
                first_two_lines = '\n'.join(cleaned_content.split('\n')[:2]).lower()

                if any(marker in first_two_lines for marker in summary_markers):
                    logger.warning("Summary markers detected in first two lines. Using original content.")
                    return document

                # Check if content structure is preserved
                orig_sections = len([l for l in document.page_content.split('\n') if l.strip().startswith('#')])
                clean_sections = len([l for l in cleaned_content.split('\n') if l.strip().startswith('#')])

                if abs(orig_sections - clean_sections) > 1:  # Allow for 1 section difference
                    logger.warning(f"Document structure changed (sections: {orig_sections} -> {clean_sections}). Using original content.")
                    return document

                # Create new document with cleaned content but preserve metadata
                cleaned_document = Document(
                    page_content=cleaned_content,
                    metadata={
                        **document.metadata,
                        "cleaned": True,
                        "cleaning_model": self.model,
                        "cleaning_provider": self.llm_provider,
                        "original_length": len(document.page_content),
                        "cleaned_length": len(cleaned_content)
                    }
                )

                reduction_percent = ((len(document.page_content) - len(cleaned_content)) / len(document.page_content)) * 100
                logger.debug(f"Document cleaned and think tags removed: {len(document.page_content)} -> {len(cleaned_content)} chars ({reduction_percent:.1f}% reduction)")
                return cleaned_document
            else:
                logger.warning("No response from LLM during cleaning, returning original document")
                return document

        except Exception as e:
            logger.error(f"Error cleaning document: {e}")
            logger.warning("Returning original document due to cleaning error")
            return document

    def clean(self, documents: List[Document], cleaning_prompt: Optional[str] = None) -> List[Document]:
        """Clean multiple documents

        Args:
            documents: List of documents to clean
            cleaning_prompt: Custom cleaning prompt (optional)

        Returns:
            List of cleaned documents
        """
        logger.info(f"Cleaning {len(documents)} documents with {self.llm_provider}/{self.model}")

        cleaned_documents = []
        total_original_chars = 0
        total_cleaned_chars = 0

        for i, doc in enumerate(documents):
            logger.debug(f"Cleaning document {i+1}/{len(documents)}")
            cleaned_doc = self.clean_document(doc, cleaning_prompt)
            cleaned_documents.append(cleaned_doc)

            total_original_chars += len(doc.page_content)
            total_cleaned_chars += len(cleaned_doc.page_content)

        reduction_percent = ((total_original_chars - total_cleaned_chars) / total_original_chars) * 100 if total_original_chars > 0 else 0
        logger.info(f"Successfully cleaned {len(cleaned_documents)} documents: {total_original_chars} -> {total_cleaned_chars} chars ({reduction_percent:.1f}% reduction)")
        return cleaned_documents

    def _get_default_cleaning_prompt(self) -> str:
        """Default cleaning prompt"""
        return """⚠️ CRITICAL INSTRUCTION: You are a document cleaner that MUST return the EXACT original Markdown document, removing ONLY navigation and UI elements.

❌ DO NOT UNDER ANY CIRCUMSTANCES:
- Create summaries or overviews
- Start with "Here's" or "In this article"
- Add introductory text
- Modify ANY technical content
- Change document structure
- Add or remove paragraphs
- Rewrite or rephrase text
- Change formatting
- Add line breaks
- Add or remove sections
- Improve or clarify text

✅ YOUR ONLY JOB IS TO:
1. Remove these Markdown elements if they appear to be UI/navigation:
   - Navigation link sections (e.g., "[Home] [About] [Contact]")
   - Footer sections with site links
   - Social media link sections
   - Newsletter subscription sections
   - Advertisement sections
   - "Related Posts" or "See Also" sections
2. Keep EVERYTHING else EXACTLY as is

📋 EXAMPLES OF FORBIDDEN OUTPUTS - DO NOT DO THESE:

❌ WRONG:
"Here's what this article covers..."
"In this technical guide, we'll explore..."
"This document explains how to..."
"Key points covered include..."
"Overview of the topic..."

✅ CORRECT:
Return the EXACT original content, just without UI elements.

IF YOU CREATE A SUMMARY, YOU HAVE FAILED.
IF YOU ADD INTRODUCTORY TEXT, YOU HAVE FAILED.
IF YOU MODIFY TECHNICAL CONTENT, YOU HAVE FAILED.

Example input:
[Home](/home) | [Docs](/docs) | [About](/about)

# Technical Guide

_Share this article: [Twitter] [LinkedIn] [Facebook]_

## Setup Instructions
1. Install dependencies:
```bash
npm install package
```

## Configuration
Edit config.json...

---
Related articles:
- [Getting Started Guide](/start)
- [Advanced Configuration](/config)
---
Copyright 2024 | [Terms](/terms) | [Privacy](/privacy)

Example output:
# Technical Guide

## Setup Instructions
1. Install dependencies:
```bash
npm install package
```

## Configuration
Edit config.json...

⚠️ FINAL WARNING:
- NO summaries
- NO overviews
- NO introductions
- NO modifications
- ONLY remove UI elements
- Keep ALL technical content EXACTLY as is"""


def create_document_cleaner(llm_provider: str, model: str) -> DocumentCleaner:
    """Factory function to create a document cleaner

    Args:
        llm_provider: LLM provider (ollama, openai, gemini)
        model: Model name

    Returns:
        DocumentCleaner instance
    """
    # Validate provider
    valid_providers = ["ollama", "openai", "gemini"]
    if llm_provider not in valid_providers:
        raise ValueError(f"Invalid LLM provider: {llm_provider}. Must be one of: {valid_providers}")

    # Validate model name is not empty
    if not model or not model.strip():
        raise ValueError("Model name cannot be empty")

    return DocumentCleaner(llm_provider, model.strip())


def clean_documents(
    documents: List[Document],
    enable_cleaning: bool,
    llm_provider: str,
    model: str,
    cleaning_prompt: Optional[str] = None
) -> List[Document]:
    """Clean documents if cleaning is enabled

    Args:
        documents: Documents to potentially clean
        enable_cleaning: Whether to enable cleaning
        llm_provider: LLM provider for cleaning
        model: Model name for cleaning
        cleaning_prompt: Custom cleaning prompt (optional)

    Returns:
        Cleaned documents if enabled, otherwise original documents
    """
    if not enable_cleaning:
        logger.debug("Document cleaning disabled, skipping")
        return documents

    if not documents:
        logger.debug("No documents to clean")
        return documents

    try:
        cleaner = create_document_cleaner(llm_provider, model)
        return cleaner.clean(documents, cleaning_prompt)
    except Exception as e:
        logger.error(f"Failed to initialize document cleaner: {e}")
        logger.warning("Falling back to original documents without cleaning")
        return documents
