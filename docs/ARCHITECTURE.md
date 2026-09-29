# OpsMind Architecture

OpsMind is an end-to-end incident-intelligence application. The same codebase covers document ingestion, supervised classification, semantic retrieval, grounded RAG, API serving, evaluation, and engineering automation.

## System diagram

```mermaid
flowchart TD
    U[User / Client] --> API[FastAPI / CLI]

    API --> DP[Document Processing]
    DP --> DISC[Discovery]
    DISC --> LOAD[TXT / Markdown Loaders]
    LOAD --> CLEAN[Cleaner + Processor]
    CLEAN --> PD[ProcessedDocument]

    PD --> CL{Classification}
    CL --> C1[TF-IDF + Logistic Regression]
    CL --> C2[PyTorch Embedding + MLP]
    CL --> C3[DistilBERT Classifier]
    C1 --> CP[IncidentPrediction]
    C2 --> CP
    C3 --> CP

    KB[Runbook Knowledge Base] --> DP2[Document Processing]
    DP2 --> CH[Chunking + Overlap]
    CH --> EMB[Sentence-Transformer Embeddings]
    EMB --> QD[(Embedded Qdrant Vector Store)]

    API --> Q[User Question]
    Q --> QE[Query Embedding]
    QE --> QD
    QD --> TOP[Top-K Relevant Chunks]
    TOP --> PROMPT[Grounded Prompt]
    PROMPT --> GEN{Generator}
    GEN --> HF[Local FLAN-T5]
    GEN --> EX[Extractive Fallback]
    GEN --> OA[OpenAI-compatible API]
    HF --> RA[RAG Answer + Sources]
    EX --> RA
    OA --> RA

    CP --> REP[Reporting / API Response]
    RA --> REP
```

## Core boundaries

- **Document layer:** discovers, loads, cleans, validates, and normalizes operational documents.
- **Classification layer:** exposes three independent approaches behind the same `IncidentPrediction` contract.
- **Retrieval layer:** chunks runbooks, embeds text, persists vectors and metadata in embedded Qdrant, and performs semantic Top-K search.
- **RAG layer:** combines retrieved evidence with a generator and keeps source chunks attached to the answer.
- **API layer:** exposes health, classification, retrieval, and RAG through HTTP while caching heavyweight services per process.
- **Evaluation layer:** measures retrieval independently from generation so failures can be localized.
- **Engineering layer:** package metadata, validated environment configuration, automated tests, GitHub preflight checks, and GitHub Actions CI.

## Main request flows

### Classification

```text
HTTP /v1/classify or CLI
        ↓
ProcessedDocument
        ↓
Selected classifier
        ↓
IncidentPrediction
        ↓
JSON response/report
```

### Retrieval

```text
Question
   ↓
Sentence embedding (384 dimensions)
   ↓
Embedded Qdrant cosine search
   ↓
Top-K chunks + metadata + scores
```

### RAG

```text
Question
   ↓
Semantic retrieval
   ↓
Top-K runbook chunks
   ↓
Grounded prompt
   ↓
Generator
   ↓
Answer + source chunks
```

## Local runtime

The portfolio release intentionally uses a lightweight local runtime:

```text
Windows / macOS / Linux
        ↓
Python virtual environment
        ↓
OpsMind application
        ├── FastAPI / CLI
        ├── local Hugging Face models
        └── embedded persistent Qdrant
```

A separate vector-database service is not required for the documented local workflow.
