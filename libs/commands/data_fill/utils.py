import argparse
import re
import logging
from bs4 import BeautifulSoup
from typing import List
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_core.documents import Document
from ruamel.yaml import YAML
from io import StringIO

logger = logging.getLogger("RAG")

tags_to_remove = [
    "ad",
    "advertisement",
    "aside",
    "banner",
    "comments",
    "complementary",
    "contentinfo",
    "cookie",
    "dialog",
    "footer",
    "form",
    "head",
    "header",
    "iframe",
    "menu",
    "menubar",
    "modal",
    "nav",
    "newsletter",
    "noscript",
    "popup",
    "promotion",
    "recommended",
    "related",
    "script",
    "search",
    "share",
    "sidebar",
    "social",
    "style",
    "subscribe",
    "template",
    "toolbar",
    "widget",
]


def log_data_fill_options(args: argparse.Namespace) -> None:
    """Log data fill options for user feedback."""
    if args.cleanup:
        logger.info("Cleanup enabled: collection will be deleted before filling.")
    if args.clean_content:
        logger.info(
            "Document cleaning enabled: HTML tags and UI elements will be removed before Markdown conversion."
        )
    else:
        logger.info(
            "Document cleaning disabled: raw HTML will be converted to Markdown without pre-cleaning."
        )
    if args.extract_wisdom:
        logger.info(
            f"Wisdom extraction enabled: {args.fabric_command} will be used to extract key insights."
        )


def sanitize_filename(title: str) -> str:
    # Replace spaces and special characters with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", title)
    # Convert to lowercase
    sanitized = sanitized.lower()
    # Replace multiple underscores with single one
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized


def extract_title_from_html(content: str) -> str:
    soup = BeautifulSoup(content, "html.parser")
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        return title_tag.get_text(strip=True)
    h1_tag = soup.find("h1")
    if h1_tag and h1_tag.get_text(strip=True):
        return h1_tag.get_text(strip=True)
    return "untitled"


def process_html_documents(
    docs: List[Document], clean_content: bool = False
) -> List[Document]:
    if clean_content:
        total_original = sum(len(doc.page_content) for doc in docs)
        cleaned_docs = []
        for doc in docs:
            cleaned_html = clean_html_content(doc.page_content)
            doc.page_content = cleaned_html
            cleaned_docs.append(doc)
        total_cleaned = sum(len(doc.page_content) for doc in cleaned_docs)
        reduction_percent = (
            ((total_original - total_cleaned) / total_original * 100)
            if total_original > 0
            else 0
        )
        logger.info(
            f"HTML tags and UI elements removed before Markdown conversion. Content reduced from {total_original} to {total_cleaned} chars ({reduction_percent:.1f}% reduction)."
        )
    else:
        logger.info(
            "Document cleaning disabled: raw HTML will be converted to Markdown without pre-cleaning."
        )
        cleaned_docs = docs

    return cleaned_docs


def convert_to_markdown(docs: List[Document]) -> List[Document]:
    logger.debug("Configuring Markdownify transformer")
    md = MarkdownifyTransformer(
        strip=[
            "script",
            "style",
            "meta",
            "link",
            "iframe",
            "button",
            "input",
            "select",
            "textarea",
        ],
        remove=tags_to_remove,
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
        keep_formatting=True,
    )
    return md.transform_documents(docs)


def clean_html_content(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    # Remove unwanted tags typical for online publications
    for tag in soup.find_all(tags_to_remove):
        tag.decompose()
    # Keep only main content tags if present
    main_content = None
    for main_tag in ["article", "main"]:
        main_content = soup.find(main_tag)
        if main_content:
            break
    if main_content:
        return str(main_content)
    return str(soup)


def metadata_to_yaml(metadata: dict) -> str:
    """Convert metadata dict to YAML frontmatter using ruamel.yaml for robust escaping."""
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.allow_unicode = True
    stream = StringIO()
    yaml.dump(metadata, stream)
    yaml_content = stream.getvalue()
    return f"---\n{yaml_content}---\n"


def format_content(
    doc: Document,
    wisdom: str,
) -> Document:
    new_content = ""
    if len(doc.metadata) > 0:
        new_content += metadata_to_yaml(doc.metadata)
    if len(wisdom) > 0:
        new_content += f"""
{wisdom}

---

        """
    new_content += f"\n{doc.page_content}"
    doc.page_content = new_content
    return doc


def parse_source_with_title(source_path: str):
    """Split source_path on '||' to extract (source, title)."""
    if '||' in source_path:
        source, title = source_path.split('||', 1)
        return source.strip(), title.strip()
    return source_path.strip(), None
