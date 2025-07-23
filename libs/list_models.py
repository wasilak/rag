import logging
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from .models import get_model_manager
import humanize

logger = logging.getLogger("RAG")
console = Console()


def process_list_models(provider: str) -> None:
    """Process the list-models command

    Args:
        provider: LLM provider to list models for (ollama, openai, gemini)
    """
    logger.info(f"Listing available models for {provider}")

    try:
        manager = get_model_manager()
        models = manager.list_models(provider)

        if not models:
            console.print(f"[red]No models found for {provider} or unable to connect[/red]")
            return

        # Display models in a nice table format
        _display_models_table(provider, models)

        # Show default models
        _display_default_models(provider, manager)

    except Exception as e:
        logger.error(f"Error listing models for {provider}: {e}")
        console.print(f"[red]Error: {e}[/red]")


def _display_models_table(provider: str, models: List[Dict]) -> None:
    """Display models in a formatted table"""

    table = Table(title=f"Available Models for {provider.title()}")

    if provider == "ollama":
        table.add_column("Name", style="cyan")
        table.add_column("Size", style="magenta")

        for model in models:
            size = humanize.naturalsize(model.get('size', 0), binary=True)
            table.add_row(
                model.get('name', 'N/A'),
                size,
            )

    elif provider == "openai":
        table.add_column("ID", style="cyan")
        table.add_column("Owner", style="magenta")

        for model in models:
            table.add_row(
                model.get('id', 'N/A'),
                model.get('owned_by', 'N/A'),
            )

    elif provider == "gemini":
        table.add_column("Name", style="cyan")
        table.add_column("Capabilities", style="yellow")

        for model in models:
            capabilities = ', '.join(model.get('supported_generation_methods', []))
            table.add_row(
                model.get('name', 'N/A'),
                capabilities
            )

    console.print(table)
    console.print(f"\n[bold]Total models found: {len(models)}[/bold]")


def _display_default_models(provider: str, manager) -> None:
    """Display default models for the provider"""
    try:
        chat_default = manager.get_default_model(provider, "chat")
        embedding_default = manager.get_default_model(provider, "embedding")

        defaults_panel = Panel(
            f"[bold]Chat Model:[/bold] {chat_default}\n[bold]Embedding Model:[/bold] {embedding_default}",
            title=f"Default Models for {provider.title()}",
            border_style="blue"
        )
        console.print(defaults_panel)

    except Exception as e:
        logger.debug(f"Could not display defaults for {provider}: {e}")
