from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qdrant_client")

from opsmind.models.retrieval import DocumentChunk
from opsmind.retrieval.vector_store import QdrantVectorStore


def test_qdrant_vector_store_round_trip(
    tmp_path: Path,
) -> None:
    store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_chunks",
    )

    chunks = (
        DocumentChunk(
            chunk_id="11111111-1111-1111-1111-111111111111",
            source_path=Path("database.md"),
            title="Database",
            chunk_index=0,
            text="database timeout connection pool",
            word_count=4,
        ),
        DocumentChunk(
            chunk_id="22222222-2222-2222-2222-222222222222",
            source_path=Path("network.md"),
            title="Network",
            chunk_index=0,
            text="packet loss network route",
            word_count=4,
        ),
    )

    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    try:
        store.rebuild(
            chunks=chunks,
            vectors=vectors,
            vector_dimension=2,
        )

        hits = store.search(
            query_vector=np.asarray(
                [1.0, 0.0],
                dtype=np.float32,
            ),
            top_k=1,
        )
    finally:
        store.close()

    assert len(hits) == 1
    assert hits[0].source_path == Path("database.md")
