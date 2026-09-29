"""Pure text-cleaning functions."""


def normalize_title(title: str) -> str:
    """Trim a title and collapse repeated whitespace."""

    return " ".join(title.split())


def clean_text(text: str) -> str:
    """Normalize line endings, whitespace, and empty lines."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = [
        " ".join(line.split())
        for line in normalized.split("\n")
    ]

    non_empty_lines = [
        line for line in cleaned_lines if line
    ]

    return "\n".join(non_empty_lines)


def count_words(text: str) -> int:
    """Count whitespace-separated words."""

    return len(text.split())
