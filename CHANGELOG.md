# Changelog

## 1.0.2 — Portfolio release cleanup

- Removed unverified deployment tooling from the public portfolio scope.
- Standardized the documented local runtime around Python, FastAPI, local Hugging Face models, and embedded Qdrant.
- Updated README, architecture, resume, interview, validation, and publishing documentation.
- Added a GitHub preflight script to prevent secrets, model artifacts, runtime vector storage, generated reports, backups, and large files from being committed accidentally.
- Kept GitHub Actions CI as the verified automation target.

## 1.0.1 — Qdrant/RAG resource lifecycle hotfix

- Shared one retrieval service/Qdrant client across `/v1/retrieve` and `/v1/ask`.
- Added regression coverage for shared retrieval ownership.
- Fixed the local embedded-Qdrant double-open failure observed after retrieval followed by RAG in the same API process.

## 1.0.0 — Feature-complete AI application

- Document ingestion and processing pipeline.
- Classical, PyTorch, and DistilBERT incident classification.
- Sentence-Transformer embeddings, Qdrant semantic retrieval, and retrieval evaluation.
- Grounded RAG with local, extractive, and OpenAI-compatible generator adapters.
- FastAPI serving, structured configuration, reporting, logging, and automated tests.
