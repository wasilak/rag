import logging
from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from .models import get_model_manager, ModelManager
import humanize

logger = logging.getLogger("RAG")
console = Console()

# Cache for models
_cached_models: Dict[str, List[Dict]] = {}
_model_manager: Optional[ModelManager] = None


def initialize_model_manager(host: str = "127.0.0.1", port: int = 11434) -> None:
    """Initialize the model manager singleton.

    Args:
        host: Ollama host address
        port: Ollama port number
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = get_model_manager(host, port)


def get_cached_models(provider: str) -> Optional[List[Dict]]:
    """Get cached models for a provider.

    Args:
        provider: LLM provider name

    Returns:
        Cached models list if available, None otherwise
    """
    return _cached_models.get(provider)


def process_list_models(
    provider: str, force_refresh: bool = False, silent: bool = False,
    ollama_host: str = "127.0.0.1", ollama_port: int = 11434
) -> Optional[List[Dict]]:
    """Process the list-models command

    Args:
        provider: LLM provider to list models for (ollama, openai, gemini)
        force_refresh: Force refresh of cached models
        silent: If True, don't display tables/output (for pre-caching)

    Returns:
        List of models if successful, None otherwise
    """
    # Check cache first unless force refresh requested
    if not force_refresh and provider in _cached_models:
        logger.debug(f"Using cached models for {provider}")
        models = _cached_models[provider]
        if not silent:
            _display_models_table(provider, models)
            _display_default_models(provider, _model_manager)
        return models

    if not silent:
        logger.info(f"Listing available models for {provider}")
    else:
        logger.debug(f"Verifying models for {provider}")

    try:
        initialize_model_manager(ollama_host, ollama_port)
        if _model_manager is None:
            console.print("[red]Failed to initialize model manager[/red]")
            return None

        models = _model_manager.list_models(provider)

        if not models:
            if not silent:
                console.print(
                    f"[red]No models found for {provider} or unable to connect[/red]"
                )
            else:
                logger.warning(f"No models found for {provider} or unable to connect")
            return None

        # Cache the models
        _cached_models[provider] = models

        if not silent:
            # Display models in a nice table format
            _display_models_table(provider, models)

            # Show default models
            _display_default_models(provider, _model_manager)
        else:
            logger.debug(f"Successfully verified {len(models)} models for {provider}")

        return models

    except Exception as e:
        logger.error(f"Error listing models for {provider}: {e}")
        if not silent:
            console.print(f"[red]Error: {e}[/red]")
        return None


def _display_models_table(provider: str, models: List[Dict]) -> None:
    """Display models in a formatted table"""

    table = Table(title=f"Available Models for {provider.title()}")

    if provider == "ollama":
        table.add_column("Name", style="cyan")
        table.add_column("Size", style="magenta")

        for model in models:
            size = humanize.naturalsize(model.get("size", 0), binary=True)
            table.add_row(
                model.get("name", "N/A"),
                size,
            )

    elif provider == "openai":
        table.add_column("ID", style="cyan")
        table.add_column("Owner", style="magenta")

        for model in models:
            table.add_row(
                model.get("id", "N/A"),
                model.get("owned_by", "N/A"),
            )

    elif provider == "gemini":
        table.add_column("Name", style="cyan")

        for model in models:
            table.add_row(
                model.get("name", "N/A"),
            )

    console.print(table)
    console.print(f"\n[bold]Total models found: {len(models)}[/bold]")


def _display_default_models(
    provider: str, manager: Optional[ModelManager] = None
) -> None:
    """Display default models for the provider"""
    if manager is None:
        return

    try:
        chat_default = manager.get_default_model(provider, "chat")
        embedding_default = manager.get_default_model(provider, "embedding")

        defaults_panel = Panel(
            f"[bold]Chat Model:[/bold] {chat_default}\n[bold]Embedding Model:[/bold] {embedding_default}",
            title=f"Default Models for {provider.title()}",
            border_style="blue",
        )
        console.print(defaults_panel)

    except Exception as e:
        logger.debug(f"Could not display defaults for {provider}: {e}")
