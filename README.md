# UniRAG — Transparent & Automated University Application Evaluation

An end-to-end **RAG platform** for evaluating university applications using **locally hosted LLMs**, built with FAISS, LangChain-style modular pipelines, and SentenceTransformers — designed around **GDPR** and **EU AI Act** principles.

## Architecture

```
┌─────────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│ Synthetic Data  │──▶│  Ingestion   │──▶│   Chunking    │──▶│  Embeddings  │
│ (Faker: SOPs,   │   │  + PII       │   │  (semantic /  │   │ (Sentence-   │
│  LORs, CVs,     │   │  Anonymizer  │   │   recursive)  │   │ Transformers)│
│  transcripts)   │   └──────────────┘   └───────────────┘   └──────┬───────┘
└─────────────────┘                                                 ▼
┌─────────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│  Streamlit UI / │◀──│  Evaluator   │◀──│   Retriever   │◀──│ FAISS Vector │
│  Audit Log      │   │ (local LLM + │   │ (top-k + MMR, │   │    Store     │
│  (transparency) │   │  rubric      │   │  per-applicant│   │  + metadata  │
└─────────────────┘   │  prompts)    │   │  filtering)   │   └──────────────┘
                      └──────────────┘   └───────────────┘
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Generate synthetic applicant data (no real PII — GDPR-safe by construction)
python data_gen/generate_applicants.py --n 25

# 2. Build the FAISS index
python -m src.pipeline build-index

# 3. Evaluate an applicant (requires Ollama running: `ollama pull llama3.2`)
python -m src.pipeline evaluate --applicant APP-0001

# 4. Launch the UI
streamlit run app/app.py
```

## Local LLM

Uses [Ollama](https://ollama.com) by default (`llama3.2`, a quantized SLM — fast on CPU).
Swap models in `config.yaml`. Everything runs locally: **no applicant data leaves your machine**.

## GDPR / EU AI Act design notes

- **Data minimization**: only fields needed for evaluation are ingested.
- **Pseudonymization**: `src/privacy/anonymizer.py` strips names/emails before indexing; evaluation uses applicant IDs.
- **Local processing**: embeddings + LLM inference are fully on-device.
- **Transparency (AI Act, high-risk system)**: every evaluation stores retrieved evidence chunks, the prompt, model ID, and rubric scores in `data/audit/` — a human reviewer can trace every claim.
- **Human oversight**: the system outputs a *recommendation*, never a final decision.

## Repo layout

```
data_gen/            synthetic applicant generator (Faker)
src/ingestion/       loaders + document models
src/privacy/         PII pseudonymization
src/embeddings/      SentenceTransformers wrapper
src/vectorstore/     FAISS store with metadata + persistence
src/retrieval/       top-k retrieval with applicant filtering
src/llm/             Ollama client (swappable)
src/orchestration/   rubric prompts + evaluator + audit logging
app/                 Streamlit reviewer UI
```
