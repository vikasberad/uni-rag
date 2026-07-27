"""End-to-end pipeline CLI.

  python -m src.pipeline build-index
  python -m src.pipeline evaluate --applicant APP-0001
  python -m src.pipeline query --q "strong research background in NLP"
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

app = typer.Typer(add_completion=False)
CFG = yaml.safe_load(Path("config.yaml").read_text())


def _embedder():
    from src.embeddings.embedder import Embedder
    e = CFG["embeddings"]
    return Embedder(e["model_name"], e["batch_size"])


@app.command("build-index")
def build_index():
    from src.ingestion.loader import load_all
    from src.ingestion.chunker import chunk_documents
    from src.privacy.anonymizer import pseudonymize
    from src.vectorstore.faiss_store import FaissStore

    embedder = _embedder()
    store = FaissStore(embedder.dim)
    c = CFG["chunking"]

    for profile, docs in load_all(CFG["data"]["raw_dir"]):
        if CFG["privacy"]["pseudonymize"]:
            for d in docs:
                d.text = pseudonymize(d.text, profile)
        chunks = chunk_documents(docs, c["chunk_size"], c["chunk_overlap"])
        vecs = embedder.embed([ch.text for ch in chunks])
        metas = [{"chunk_id": ch.chunk_id, "applicant_id": ch.applicant_id,
                  "doc_type": ch.doc_type, "text": ch.text} for ch in chunks]
        store.add(vecs, metas)
        typer.echo(f"Indexed {profile['applicant_id']}: {len(chunks)} chunks")

    store.save(CFG["data"]["index_dir"])
    typer.echo(f"Index saved -> {CFG['data']['index_dir']} ({store.index.ntotal} vectors)")


def _retriever():
    from src.vectorstore.faiss_store import FaissStore
    from src.retrieval.retriever import Retriever
    store = FaissStore.load(CFG["data"]["index_dir"])
    return Retriever(store, _embedder(), top_k=CFG["retrieval"]["top_k"])


@app.command()
def query(q: str = typer.Option(...), applicant: str = typer.Option(None)):
    for r in _retriever().retrieve(q, applicant):
        typer.echo(f"[{r['score']:.3f}] {r['applicant_id']}/{r['doc_type']}: "
                   f"{r['text'][:120]}...")


@app.command()
def evaluate(applicant: str = typer.Option(...)):
    from src.privacy.anonymizer import minimize_profile
    from src.llm.local_llm import get_llm
    from src.orchestration.evaluator import evaluate_applicant

    profile = json.loads(
        (Path(CFG["data"]["raw_dir"]) / applicant / "profile.json").read_text())
    out = evaluate_applicant(
        applicant, minimize_profile(profile), _retriever(),
        get_llm(CFG["llm"]), CFG["data"]["audit_dir"])
    typer.echo(json.dumps(out["result"], indent=2))
    typer.echo(f"\nAudit trail: {out['audit_file']}")


if __name__ == "__main__":
    app()
