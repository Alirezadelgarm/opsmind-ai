# OpsMind Interview Guide

## 30-second explanation

OpsMind is an end-to-end AI engineering project for operational incident intelligence. It ingests and normalizes incident documents, compares three text-classification approaches, indexes operational runbooks with sentence embeddings in embedded Qdrant, retrieves semantically relevant chunks, generates grounded RAG answers with source evidence, and exposes the system through FastAPI. The repository also contains explicit evaluation, automated tests, configuration management, Git-safe artifact handling, and GitHub Actions CI.

## Why three classifiers?

They demonstrate the trade-off between a strong small-data classical baseline, a from-scratch PyTorch NLP model, and transfer learning with a pretrained Transformer. Keeping all three provides an engineering comparison rather than assuming the most complex model is automatically best.

## Why Qdrant?

Retrieval requires more than storing vectors in a Python list. Qdrant gives the project persistence, metadata payloads, cosine search, and a clean vector-store abstraction. The portfolio release uses embedded local Qdrant so the complete retrieval/RAG path works without running a separate database service.

## Why evaluate retrieval separately from RAG?

If retrieval is wrong, the generator never receives the evidence. If retrieval is correct but the answer is poor, the generator or prompt is the bottleneck. Separate metrics make the failure location measurable.

## Why FastAPI?

The ML/RAG logic becomes a reusable network service instead of code that only works inside a Python process or CLI. Other applications can call `/v1/classify`, `/v1/retrieve`, and `/v1/ask` over HTTP.

## What CI is implemented?

GitHub Actions installs the project on Python 3.11 and 3.13, checks dependency consistency, compiles the source, and runs the automated tests on pushes and pull requests. The repository implements CI, not automatic production deployment.

## What is intentionally out of scope?

The documented release is aligned with what was actually verified: local Python execution, embedded Qdrant, FastAPI, tests, and CI. Automated production deployment is not part of v1.0.2, so the project does not claim a full CI/CD pipeline.

## A bug worth discussing

The initial FastAPI RAG path opened a second embedded Qdrant client against the same local storage directory after `/v1/retrieve` had already opened it. OpsMind v1.0.1 fixed this by sharing one retrieval service/Qdrant client across API endpoints and adding a regression test. This is a useful example of resource lifecycle management in a real AI application.

## Current limitation worth stating honestly

The retrieval fixture performs strongly on a tiny curated set, but the RAG generation evaluation has lower keyword recall. A stronger generator, prompt experiments, a larger evaluation set, and calibrated acceptance criteria would be sensible next steps before production use.
