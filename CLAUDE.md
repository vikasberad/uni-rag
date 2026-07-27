# CLAUDE.md — project context for Claude Code

## What this is
RAG platform for transparent university application evaluation. Synthetic applicants
(Faker) → pseudonymization → chunking → SentenceTransformers embeddings → FAISS →
rubric-driven retrieval → local LLM (Ollama) → JSON evaluation + audit trail → Streamlit UI.

## Commands
- Generate data: `python data_gen/generate_applicants.py --n 25`
- Build index: `python -m src.pipeline build-index`
- Evaluate: `python -m src.pipeline evaluate --applicant APP-0001`
- Search: `python -m src.pipeline query --q "..." [--applicant APP-0001]`
- UI: `streamlit run app/app.py`
- Tests: `pytest -q`

## Conventions
- All config lives in `config.yaml`; never hardcode paths or model names.
- PII never enters the index or prompts: pseudonymize via `src/privacy/anonymizer.py`,
  minimize profiles via `minimize_profile()`. Identity mapping stays only in profile.json.
- Every LLM evaluation must write a full audit record to `data/audit/`.
- LLM output must be strict JSON (see `SYSTEM_PROMPT` in `src/orchestration/evaluator.py`).
- Local-only: no external API calls for embeddings or inference.

## Roadmap (good next tasks)
1. Hybrid retrieval: add BM25 (rank_bm25) + reciprocal rank fusion in `src/retrieval/`.
2. Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
3. Agentic flow: multi-step evaluator agent (plan → retrieve per criterion → score → verify citations).
4. MCP server: expose `retrieve` and `evaluate` as MCP tools (Python `mcp` SDK) so any MCP client can call the pipeline.
5. Batch evaluation + calibration report against the hidden `_strength` label in profiles.
6. Swap FAISS for Qdrant/Chroma behind the same `FaissStore` interface (compare recall/latency).
7. PDF ingestion (pypdf) so real document formats work.
