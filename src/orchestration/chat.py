"""Conversational RAG engine for the reviewer chat.

A professor can ask free-form questions about one applicant, compare several,
or query the whole pool. Answers are grounded ONLY in retrieved evidence
chunks and minimized profiles; every turn is written to the audit trail.

Design notes (thesis-relevant):
- Scope control: the UI passes explicit applicant IDs; additionally any
  APP-XXXX mentioned in the question is auto-added to the scope.
- Grounding: evidence chunks are labeled [E1..En] and the model is instructed
  to cite them and to say clearly when evidence is missing.
- Comparison: for multi-applicant scope we retrieve per applicant so no
  candidate dominates the context.
- Transparency: prompt, evidence, model ID and answer are logged per turn.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

APP_ID_RE = re.compile(r"APP-\d{4}", re.IGNORECASE)

CHAT_SYSTEM_PROMPT = """You are an admissions review assistant for Master's programs at the
University of Stuttgart (MSc Computer Science, MSc INFOTECH, MSc Electrical Engineering).
You help a professor explore and compare applicants. You SUPPORT the human reviewer —
you never make or imply a final admission decision.

Rules:
1. Base every claim ONLY on the provided evidence chunks [E1..En] and the structured
   profiles. Never invent facts about an applicant.
2. Cite evidence IDs inline, e.g. "strong praise from the academic referee [E3]".
3. If the evidence does not answer the question, say so explicitly instead of guessing.
4. When comparing applicants, be balanced: give each candidate's strengths and
   weaknesses, and organize the answer per applicant or per criterion.
5. Applicants are identified only by their ID (e.g. APP-0007) for privacy reasons.
6. Be concise and professional. Plain text, short paragraphs; no markdown tables.
7. The conversation history is only for understanding what the professor is asking.
   Every factual claim (project titles, scores, publications, quotes) must come from
   the CURRENT evidence chunks and profiles — never reuse details about one applicant
   from earlier turns when discussing a different applicant.
"""


def resolve_scope(question: str, selected_ids: list[str],
                  all_ids: list[str]) -> list[str]:
    """Union of UI-selected applicants and APP-XXXX ids mentioned in the question."""
    mentioned = [m.upper() for m in APP_ID_RE.findall(question)]
    scope = [a for a in dict.fromkeys(list(selected_ids) + mentioned) if a in all_ids]
    return scope


def _gather_evidence(question: str, scope: list[str], retriever,
                     per_applicant: int, pool_k: int) -> list[dict]:
    """Retrieve evidence; per-applicant when scoped, pool-wide otherwise."""
    hits: list[dict] = []
    if scope:
        for app_id in scope[:4]:  # cap to keep the prompt small for local SLMs
            hits.extend(retriever.retrieve(question, applicant_id=app_id)[:per_applicant])
    else:
        hits = retriever.retrieve(question)[:pool_k]
    for i, h in enumerate(hits, start=1):
        h["eid"] = f"E{i}"
    return hits


def _load_profiles(scope: list[str], raw_dir: str | Path, minimize) -> dict[str, dict]:
    out = {}
    for app_id in scope[:4]:
        f = Path(raw_dir) / app_id / "profile.json"
        if f.exists():
            out[app_id] = minimize(json.loads(f.read_text()))
    return out


def _format_history(history: list[dict], max_turns: int = 6) -> str:
    """Render the last turns as plain text so the model has conversational context."""
    recent = history[-max_turns:]
    lines = []
    for m in recent:
        role = "Professor" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def build_chat_prompt(question: str, profiles: dict[str, dict],
                      evidence: list[dict], history: list[dict]) -> str:
    parts = []
    if history:
        parts.append("Conversation so far:\n" + _format_history(history))
    if profiles:
        parts.append("Structured profiles (pseudonymized):\n"
                     + json.dumps(profiles, indent=2))
    if evidence:
        ev = "\n\n".join(
            f"[{h['eid']}] (applicant {h['applicant_id']}, source: {h['doc_type']})\n{h['text']}"
            for h in evidence)
        parts.append("Evidence chunks:\n\n" + ev)
    else:
        parts.append("Evidence chunks: NONE FOUND for this question.")
    parts.append(f"Professor's question: {question}\n\n"
                 "Answer the question following the rules. Cite evidence IDs.")
    return "\n\n".join(parts)


def chat_turn(question: str, selected_ids: list[str], all_ids: list[str],
              retriever, llm, raw_dir: str | Path, audit_dir: str | Path,
              minimize, history: list[dict] | None = None,
              fallback_scope: list[str] | None = None,
              per_applicant: int = 4, pool_k: int = 8) -> dict:
    """One grounded chat turn. Returns {'answer', 'evidence', 'scope', 'audit_file'}.

    fallback_scope: the previous turn's scope. If the current question mentions no
    applicant IDs and none are selected in the UI, we inherit it — so follow-ups
    like "which of the two is stronger?" stay anchored to the right applicants
    instead of degrading to pool-wide retrieval.
    """
    history = history or []
    scope = resolve_scope(question, selected_ids, all_ids)
    scope_source = "explicit"
    if not scope and fallback_scope:
        scope = [a for a in fallback_scope if a in all_ids]
        scope_source = "inherited_from_previous_turn"
    evidence = _gather_evidence(question, scope, retriever, per_applicant, pool_k)
    profiles = _load_profiles(scope, raw_dir, minimize)
    prompt = build_chat_prompt(question, profiles, evidence, history)
    answer = llm.generate(CHAT_SYSTEM_PROMPT, prompt)

    audit = {
        "type": "chat_turn",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": getattr(llm, "model", "unknown"),
        "question": question,
        "scope": scope,
        "scope_source": scope_source,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "prompt": prompt,
        "evidence": evidence,
        "answer": answer,
        "disclaimer": "AI-assisted exploration only. Final decisions require human review.",
    }
    d = Path(audit_dir)
    d.mkdir(parents=True, exist_ok=True)
    audit_file = d / f"chat_{int(time.time() * 1000)}.json"
    audit_file.write_text(json.dumps(audit, indent=2))

    return {"answer": answer, "evidence": evidence, "scope": scope,
            "audit_file": str(audit_file)}
