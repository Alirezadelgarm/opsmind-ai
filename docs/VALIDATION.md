# Release Validation Checklist

## Verified before v1.0.2 portfolio cleanup

- [x] `19 passed` automated tests on the v1.0.1 local environment
- [x] `GET /health` returned `ok`, version `1.0.1`
- [x] `POST /v1/classify` with `model_type="transformer"` returned a valid classification response
- [x] `POST /v1/retrieve` returned semantic Top-K results
- [x] `POST /v1/ask` returned a grounded answer using `hf_local`
- [x] Retrieval index built successfully: 6 documents → 13 chunks → 384-d vectors
- [x] Retrieval evaluation: Hit@1=1.0, Hit@3=1.0, MRR=1.0 on 6 curated queries
- [x] RAG evaluation: source-hit=1.0, mean keyword recall≈0.4167 on 4 curated queries

## Run after applying v1.0.2

- [ ] Run `python -m pip install -e ".[dev]"`
- [ ] Run `.\scripts\verify-local.ps1`
- [ ] Confirm `GET /health` returns version `1.0.2`
- [ ] Initialize Git, stage the repository, then run `.\scripts\github-preflight.ps1`
- [ ] Push to GitHub and confirm the `ci` workflow is green on Python 3.11 and 3.13

## Important honesty note

The evaluation datasets are deliberately small portfolio fixtures. Keep the metrics in context when discussing the project: they validate architecture and behavior; they are not production benchmarks.
