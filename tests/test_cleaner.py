from opsmind.processing.cleaner import clean_text, count_words, normalize_title


def test_normalize_title_collapses_whitespace() -> None:
    assert normalize_title("  Database   Incident  ") == "Database Incident"


def test_clean_text_normalizes_lines() -> None:
    raw = "  AI   Engineering \r\n\r\n  RAG   System  "
    assert clean_text(raw) == "AI Engineering\nRAG System"


def test_count_words() -> None:
    assert count_words("AI Engineering is useful") == 4
