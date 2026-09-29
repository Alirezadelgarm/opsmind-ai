from dataclasses import replace

from opsmind.config import load_settings
from opsmind.services import build_rag_service


class FakeRetrieval:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_rag_service_does_not_close_shared_retrieval() -> None:
    settings = replace(load_settings(), llm_provider="extractive")
    retrieval = FakeRetrieval()

    service = build_rag_service(
        settings,
        retrieval_service=retrieval,  # type: ignore[arg-type]
    )
    service.close()

    assert retrieval.closed is False
