import logging
import shutil
import subprocess
import re

logger = logging.getLogger("RAG")


def check_fabric_installed(command: str = "fabric") -> bool:
    return shutil.which(command) is not None


def extract_wisdom(content: str, fabric_command: str = "fabric") -> str:
    try:
        # Echo content to Fabric through stdin
        result = subprocess.run(
            [fabric_command, "-p", "extract_wisdom"], input=content, capture_output=True, text=True
        )

        if result.returncode == 0 and result.stdout:

            # strip ``` from the output beginning or ```markdown
            # remove it with regex
            if result.stdout.startswith("```"):
                result.stdout = re.sub(r"^```markdown?\n?", "", result.stdout)

            # strip ``` from the output end
            if result.stdout.endswith("```"):
                result.stdout = result.stdout[:-3]

            return result.stdout.strip()
        else:
            logger.warning("Fabric produced no output")
            if result.stderr:
                logger.debug(f"Fabric stderr: {result.stderr}")
            return ""

    except subprocess.CalledProcessError as e:
        logger.error(f"Fabric command failed: {e}")
        if e.stderr:
            logger.debug(f"Fabric stderr: {e.stderr}")
        return ""
    except Exception as e:
        logger.error(f"Error running Fabric: {e}")
        return ""
