"""Validation helpers module."""

import re
import logging
from pathlib import Path
from typing import Union
import validators


logger = logging.getLogger("RAG")


def validate_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    try:
        return bool(validators.url(url))
    except Exception as e:
        logger.debug(f"URL validation failed: {e}")
        return False


def validate_path(path: Union[str, Path]) -> bool:
    try:
        return Path(path).exists()
    except Exception as e:
        logger.debug(f"Path validation failed: {e}")
        return False


def validate_file(path: Union[str, Path]) -> bool:
    try:
        return Path(path).is_file()
    except Exception as e:
        logger.debug(f"File validation failed: {e}")
        return False


def validate_directory(path: Union[str, Path]) -> bool:
    try:
        return Path(path).is_dir()
    except Exception as e:
        logger.debug(f"Directory validation failed: {e}")
        return False


def validate_s3_bucket_name(bucket: str) -> bool:
    if not bucket:
        return False

    try:
        # Length between 3 and 63 characters
        if not (3 <= len(bucket) <= 63):
            return False

        # Must not start or end with a dot
        if bucket.startswith(".") or bucket.endswith("."):
            return False

        # Must be lowercase
        if bucket.lower() != bucket:
            return False

        # Must be a valid DNS name (letters, numbers, dots, and hyphens)
        if not re.match(r"^[a-z0-9][a-z0-9.-]*[a-z0-9]$", bucket):
            return False

        # Must not be formatted as an IP address
        if re.match(r"^\d+\.\d+\.\d+\.\d+$", bucket):
            return False

        # Adjacent periods not allowed
        if ".." in bucket:
            return False

        return True
    except Exception as e:
        logger.debug(f"S3 bucket name validation failed: {e}")
        return False


def validate_s3_bucket_path(path: str) -> bool:
    if not path:
        return True  # Empty path is valid

    try:
        # Remove leading/trailing slashes
        path = path.strip("/")

        # Check if path contains only valid characters
        # Letters, numbers, and common special characters
        return bool(re.match(r"^[a-zA-Z0-9!-_.*\'()/ ]+$", path))
    except Exception as e:
        logger.debug(f"S3 bucket path validation failed: {e}")
        return False


def validate_is_epub(file_path: str) -> bool:
    return file_path.lower().endswith('.epub')
