from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme
import os
import logging
import argparse
from openai import OpenAI
import tiktoken

logger = logging.getLogger("RAG")


def format_footnotes(metadatas: list[dict]) -> str:
    """
    Deduplicates and formats footnotes from a list of metadata dicts.
    Assumes each metadata contains 'sanitized_title' and 'source' fields.
    """
    seen = set()
    numbered = []
    for meta in metadatas:
        title = meta.get("sanitized_title") or meta.get("top_title") or "Untitled"
        source = meta.get("source", "unknown.md").split("/")[-1]  # Extract filename
        key = (title, source)
        if key not in seen:
            seen.add(key)
            numbered.append((title.strip(), source.strip()))

    # Format into footnotes
    result = "\n".join(
        [
            f'[{i + 1}] "{title}", `{source}`'
            for i, (title, source) in enumerate(numbered)
        ]
    )
    return result


def print_fancy_markdown(
    md: str,
    title: str,
    border_style: str = "green",
    code_theme: str = "monokai",
    borders_only: str = "all",
):
    """
    Render markdown with:
    - Styled headings
    - Syntax-highlighted code blocks
    - Wrapped in a panel for emphasis

    Args:
        borders_only: "all" for full borders, "top_bottom" for top/bottom only
    """
    # Define a default custom theme for markdown
    custom_theme = Theme(
        {
            "markdown.h1": "bold cyan",
            "markdown.h2": "bold magenta",
            "markdown.h3": "bold green",
            "markdown.code": "bright_white on dark_green",
            "markdown.block_quote": "italic yellow",
            "markdown.list_item": "white",
        }
    )

    console = Console(theme=custom_theme, highlight=True)

    md_render = Markdown(md, code_theme=code_theme)

    # Create custom border style for top/bottom only
    if borders_only == "top_bottom":
        # Use simple horizontal separators instead of full panel borders
        from rich.rule import Rule

        console.print(Rule(title, style=border_style))
        console.print(md_render)
        console.print(Rule(style=border_style))
    else:
        # Use default full borders
        console.print(
            Panel(md_render, title=title, border_style=border_style, expand=True)
        )


def create_openai_client(args: argparse.Namespace) -> OpenAI:
    """Creates an OpenAI-compatible client based on the LLM provider."""
    if args.provider == "ollama":
        logger.debug("Using Ollama as LLM")
        return OpenAI(
            base_url=f"http://{args.ollama_host}:{args.ollama_port}/v1",
            api_key="ollama",  # required, but unused
        )
    elif args.provider == "gemini":
        logger.debug("Using Gemini as LLM")
        return OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    else:
        logger.debug("Using OpenAI as LLM")
        return OpenAI()


def get_tokenizer_for_model(model_name: str):
    """Get appropriate tokenizer for the model."""
    try:
        if model_name.startswith("gpt-"):
            return tiktoken.encoding_for_model(model_name)
        elif model_name.startswith("claude-"):
            return tiktoken.get_encoding("cl100k_base")
        else:
            return tiktoken.encoding_for_model("gpt-4")  # Default fallback
    except Exception:
        logger.warning(f"Could not get specific tokenizer for model '{model_name}'. Falling back to GPT-4 tokenizer.")
        return tiktoken.encoding_for_model("gpt-4")
