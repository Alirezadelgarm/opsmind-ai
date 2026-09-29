# Evaluation Evidence

These results are snapshots from a local OpsMind v1.0.1 run and are retained in the v1.0.2 portfolio release under `docs/evaluation/samples/`. The v1.0.2 changes are repository/documentation cleanup and do not change the retrieval or RAG algorithms represented by these reports.

The evaluation sets are intentionally small and curated; the numbers demonstrate that the pipeline works, not production-level generalization.

## Retrieval

- Knowledge documents indexed: **6**
- Failed documents: **0**
- Chunks indexed: **13**
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector dimension: **384**
- Retrieval evaluation queries: **6**
- `Hit@1`: **1.00**
- `Hit@3`: **1.00**
- `MRR`: **1.00**

The curated retrieval set therefore placed the expected runbook at rank 1 for all six test queries. This is a useful functional signal, but six examples are not enough to claim real-world 100% retrieval accuracy.

## RAG

- RAG evaluation queries: **4**
- Source hit rate: **1.00**
- Mean keyword recall: **0.4167**
- Demonstrated local generator: `google/flan-t5-small`

The evaluation separates retrieval from answer generation. Retrieval consistently found the expected source, while generation quality was uneven. This identifies the generator/prompt/evaluation set as the next quality-improvement target rather than the vector retrieval layer.

## Why both evaluations exist

A RAG system can fail in two different places:

1. **Retrieval failure:** the relevant evidence is never supplied to the model.
2. **Generation failure:** the evidence is available, but the model omits, distorts, or poorly summarizes it.

OpsMind reports these separately so the engineering response is targeted instead of treating RAG as one opaque score.
