"""Central logging configuration."""

import logging


def configure_logging(level: str) -> None:
    """Configure application-wide logging once at startup."""

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
