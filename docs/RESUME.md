# Resume / Portfolio Notes

## Project title

**OpsMind AI — End-to-End Incident Intelligence & RAG Platform**

## Recommended resume bullets

- Built an end-to-end Python AI platform for operational incident intelligence, combining asynchronous document ingestion, three text-classification approaches (`scikit-learn`, PyTorch, and DistilBERT), semantic retrieval, grounded RAG, and REST serving through FastAPI.
- Implemented a framework-light retrieval pipeline with chunk overlap, normalized Sentence-Transformer embeddings, persistent Qdrant vector search, metadata payloads, Top-K retrieval, and explicit Hit@K/MRR evaluation.
- Designed provider-agnostic RAG with local Hugging Face generation, an extractive fallback, and an OpenAI-compatible adapter; separated retrieval evaluation from generation evaluation to localize quality bottlenecks.
- Engineered the application with environment-based configuration, model persistence, shared service lifecycle management, request/latency observability, automated pytest coverage, Git-safe artifact handling, and GitHub Actions CI across Python 3.11 and 3.13.

## Metrics you may cite with context

- 19 automated tests passed locally on v1.0.1 before the final portfolio cleanup; re-run `scripts/verify-local.ps1` on v1.0.2 before publishing.
- Retrieval fixture: 6 runbooks, 13 chunks, 384-d embeddings, Hit@1=1.00 and MRR=1.00 across 6 curated queries.
- RAG fixture: source-hit rate=1.00 with mean keyword recall≈0.4167 across 4 curated queries, highlighting generation quality as the next improvement area.

Always state that these evaluation sets are small curated fixtures, not production benchmarks.

## Claims to avoid

Do not claim production deployment, a full CI/CD pipeline, or production-grade benchmark accuracy. The repository demonstrates local application serving and CI; automated deployment is outside the current scope.
