"""MCP server exposing the UniRAG platform as callable tools.

The Model Context Protocol (MCP) is an open standard that lets any MCP-capable
client (Claude Desktop, Claude Code, an IDE, a custom agent) discover and call
tools exposed by a server. This module turns UniRAG's retrieval and evaluation
capabilities into such tools, so an external assistant can reason over the
applicant pool without ever seeing the raw documents or the FAISS index.

Design notes (thesis-relevant):
- Privacy boundary: tools return pseudonymized text and minimized profiles only.
  The ID -> identity mapping never crosses the MCP boundary.
- Audit: tool calls that invoke the LLM reuse the existing audit logging, so
  MCP-driven activity is as traceable as UI-driven activity.
- Lazy loading: the embedding model and FAISS index load on first use, so the
  server starts instantly and only pays the cost if a tool is actually called.

Transport: stdio (the client launches this process and speaks over stdin/stdout).

Run standalone:
    python -m src.mcp_server.server
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# SDK compatibility: mcp>=2.0 exposes MCPServer; mcp<2.0 exposes FastMCP.
# Both provide the same .tool() decorator and .run() entry point.
# ---------------------------------------------------------------------------
try:
    from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2.0
except ImportError:  # pragma: no cover - depends on installed SDK
    from mcp.server.fastmcp import FastMCP as _Server  # mcp < 2.0

CFG = yaml.safe_load(Path("config.yaml").read_text())
RAW_DIR = Path(CFG["data"]["raw_dir"])

mcp = _Server(
    name="unirag",
    instructions=(
        "Tools for exploring a pool of pseudonymized university applications "
        "(University of Stuttgart MSc Computer Science, Information Technology "
        "(INFOTECH), and Electrical Engineering). Applicants are identified only "
        "by IDs such as APP-0007. All evaluations are advisory: a human reviewer "
        "makes the final admission decision."
    ),
)

# ---------------------------------------------------------------------------
# Lazy singletons — built on first tool call, then reused.
# ---------------------------------------------------------------------------
_retriever = None
_llm = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from src.embeddings.embedder import Embedder
        from src.vectorstore.faiss_store import FaissStore
        from src.retrieval.retriever import Retriever
        e = CFG["embeddings"]
        _retriever = Retriever(
            FaissStore.load(CFG["data"]["index_dir"]),
            Embedder(e["model_name"], e["batch_size"]),
            top_k=CFG["retrieval"]["top_k"],
        )
    return _retriever


def get_llm():
    global _llm
    if _llm is None:
        from src.llm.local_llm import get_llm as _factory
        _llm = _factory(CFG["llm"])
    return _llm


def _all_ids() -> list[str]:
    return sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir())


def _load_profile(applicant_id: str) -> dict[str, Any]:
    return json.loads((RAW_DIR / applicant_id / "profile.json").read_text())


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    description="List every applicant ID in the pool with their target program, "
                "specialization, CGPA and German equivalent grade. Call this first "
                "to discover which applicants exist."
)
def list_applicants() -> str:
    rows = []
    for app_id in _all_ids():
        p = _load_profile(app_id)
        rows.append({
            "applicant_id": app_id,
            "program": p["program"],
            "specialization": p["intended_specialization"],
            "cgpa": p["cgpa"],
            "german_equivalent_grade": p["german_equivalent_grade"],
        })
    return json.dumps({"count": len(rows), "applicants": rows}, indent=2)


@mcp.tool(
    description="Get the pseudonymized structured profile of one applicant "
                "(grades, English test, internships, publications, interests). "
                "Personal identifiers are never returned."
)
def get_applicant_profile(applicant_id: str) -> str:
    from src.privacy.anonymizer import minimize_profile
    if applicant_id not in _all_ids():
        return json.dumps({"error": f"unknown applicant_id: {applicant_id}"})
    return json.dumps(minimize_profile(_load_profile(applicant_id)), indent=2)


@mcp.tool(
    description="Semantic search over applicant documents (SOPs, recommendation "
                "letters, CVs, transcripts). Returns the most relevant text chunks "
                "with similarity scores and their source document. Set applicant_id "
                "to search within one applicant, or leave it empty to search the "
                "whole pool."
)
def search_applications(query: str, applicant_id: str = "", top_k: int = 6) -> str:
    retriever = get_retriever()
    target = applicant_id or None
    if target and target not in _all_ids():
        return json.dumps({"error": f"unknown applicant_id: {target}"})
    hits = retriever.retrieve(query, applicant_id=target)[:top_k]
    return json.dumps({
        "query": query,
        "scope": target or "whole pool",
        "results": [
            {"applicant_id": h["applicant_id"], "source": h["doc_type"],
             "score": round(h["score"], 4), "text": h["text"]}
            for h in hits
        ],
    }, indent=2)


@mcp.tool(
    description="Retrieve the evidence chunks that support each of the five "
                "evaluation criteria (academic readiness, research potential, "
                "motivation fit, recommendations, experience and skills) for one "
                "applicant. Use this to reason about an applicant yourself without "
                "invoking the local LLM."
)
def get_evidence_for_criteria(applicant_id: str) -> str:
    if applicant_id not in _all_ids():
        return json.dumps({"error": f"unknown applicant_id: {applicant_id}"})
    evidence = get_retriever().retrieve_for_rubric(applicant_id)
    return json.dumps({
        "applicant_id": applicant_id,
        "evidence_by_criterion": {
            crit: [{"source": h["doc_type"], "score": round(h["score"], 4),
                    "text": h["text"]} for h in hits]
            for crit, hits in evidence.items()
        },
    }, indent=2)


@mcp.tool(
    description="Run the full rubric evaluation of one applicant using the local "
                "LLM. Returns 1-5 scores per criterion, an advisory recommendation "
                "(admit / borderline / reject), a justification, and the path to the "
                "audit record. This is a recommendation only; a human makes the "
                "final decision. Slow: takes 30-90 seconds on CPU."
)
def evaluate_applicant(applicant_id: str) -> str:
    from src.privacy.anonymizer import minimize_profile
    from src.orchestration.evaluator import evaluate_applicant as _evaluate
    if applicant_id not in _all_ids():
        return json.dumps({"error": f"unknown applicant_id: {applicant_id}"})
    out = _evaluate(applicant_id, minimize_profile(_load_profile(applicant_id)),
                    get_retriever(), get_llm(), CFG["data"]["audit_dir"])
    return json.dumps({
        "applicant_id": applicant_id,
        "result": out["result"],
        "evidence_count": len(out["evidence"]),
        "audit_file": out["audit_file"],
        "disclaimer": "Advisory only. Final admission decisions require human review.",
    }, indent=2)


@mcp.tool(
    description="Compare two or more applicants by returning each one's minimized "
                "profile plus balanced evidence retrieved separately per applicant, "
                "so no candidate dominates the context. Pass a comma-separated list "
                "of IDs, e.g. 'APP-0002,APP-0008'."
)
def compare_applicants(applicant_ids: str, focus: str = "overall suitability") -> str:
    from src.privacy.anonymizer import minimize_profile
    ids = [a.strip().upper() for a in applicant_ids.split(",") if a.strip()]
    known = _all_ids()
    unknown = [a for a in ids if a not in known]
    if unknown:
        return json.dumps({"error": f"unknown applicant_ids: {unknown}"})
    if len(ids) < 2:
        return json.dumps({"error": "provide at least two applicant IDs"})

    retriever = get_retriever()
    out = {"focus": focus, "applicants": {}}
    for app_id in ids[:4]:
        hits = retriever.retrieve(focus, applicant_id=app_id)[:4]
        out["applicants"][app_id] = {
            "profile": minimize_profile(_load_profile(app_id)),
            "evidence": [{"source": h["doc_type"], "score": round(h["score"], 4),
                          "text": h["text"]} for h in hits],
        }
    out["note"] = ("Evidence was retrieved separately per applicant for balance. "
                   "Base any comparison only on the returned evidence.")
    return json.dumps(out, indent=2)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
