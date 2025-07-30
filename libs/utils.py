from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme
import re


def format_footnotes(metadatas: list[dict]) -> str:
    """
    Deduplicates and formats footnotes from a list of metadata dicts.
    Assumes each metadata contains 'resolved_title' and 'source' fields.
    """
    seen = set()
    numbered = []
    for meta in metadatas:
        title = meta.get("resolved_title") or meta.get("top_title") or "Untitled"
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


def sanitize_title(title: str) -> str:
    """
    Sanitize a title by replacing special characters and spaces with underscores.
    Also ensures the title is lowercase.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", title)
    return sanitized.lower() if sanitized else "untitled"
