"""Wisdom extraction module using Fabric."""
import logging
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger("RAG")


def check_fabric_installed(command: str = 'fabric') -> bool:
    """Check if Fabric command is available in PATH

    Args:
        command: Name of the Fabric command (default: 'fabric')
    """
    return shutil.which(command) is not None


def extract_wisdom(content: str, fabric_command: str = 'fabric') -> Optional[str]:
    """Extract wisdom from content using Fabric.

    Args:
        content: Markdown content to process
        fabric_command: Name of the Fabric command (default: 'fabric')

    Returns:
        Extracted wisdom as markdown text, or None if extraction failed or Fabric not found
    """
    if not check_fabric_installed(fabric_command):
        logger.warning(f"Fabric command '{fabric_command}' not found. Skipping wisdom extraction.")
        return None

    try:
        # Echo content to Fabric through stdin
        result = subprocess.run(
            [fabric_command],
            input=content,
            capture_output=True,
            text=True
        )

        if result.returncode == 0 and result.stdout:
            return result.stdout.strip()
        else:
            logger.warning("Fabric produced no output")
            if result.stderr:
                logger.debug(f"Fabric stderr: {result.stderr}")
            return None

    except subprocess.CalledProcessError as e:
        logger.error(f"Fabric command failed: {e}")
        if e.stderr:
            logger.debug(f"Fabric stderr: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Error running Fabric: {e}")
        return None


def format_content(content: str, base_title: str, wisdom: str = "") -> tuple[str, str]:
    """Format content for files.

    Args:
        content: Original content
        base_title: Sanitized base title for link generation
        wisdom: Optional extracted wisdom

    Returns:
        Tuple of (main content, original content) where main content
        will be wisdom if provided, otherwise original content
        When wisdom is None, returns (original content, None)
    """
    if wisdom:
        # When wisdom is extracted, create both versions with cross-links
        wisdom_content = f"{wisdom.strip()}\n\n[[{base_title}_original]]"
        original_content = f"{content.strip()}\n\n[[{base_title}]]"
        return wisdom_content, original_content
    else:
        # When no wisdom, just return original content without links
        return content.strip(), ""
