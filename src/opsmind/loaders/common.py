"""Shared helpers for file loaders."""

from pathlib import Path

from opsmind.exceptions import DocumentLoadError


def read_utf8_text(path: Path) -> str:
    """Read one UTF-8 text file and translate low-level errors."""

    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DocumentLoadError(f"File not found: {path}") from exc
    except PermissionError as exc:
        raise DocumentLoadError(f"Permission denied: {path}") from exc
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            f"File is not valid UTF-8 text: {path}"
        ) from exc
    except OSError as exc:
        raise DocumentLoadError(f"Could not read file: {path}") from exc
