"""Prompt orchestration + transparent evaluation with audit trail.

Design goals (EU AI Act — treat admissions as a high-risk use case):
- Grounding: the LLM may only use retrieved evidence chunks, each labeled [E1..En].
- Traceability: full prompt, model ID, evidence, and output are written to data/audit/.
- Human oversight: output is a *recommendation* with rubric scores, never a decision.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

RUBRIC = ["academic_readiness", "research_potential", "motivation_fit",
          "recommendations", "experience_skills"]

SYSTEM_PROMPT = """You are an admissions evaluation assistant. You produce a structured,
evidence-grounded assessment to SUPPORT a human reviewer — you never make the final decision.

Rules:
1. Base every claim ONLY on the provided evidence chunks [E1..En] and the structured profile.
2. Cite evidence IDs for each criterion.
3. If evidence is missing for a criterion, say so and score conservatively.
4. Score each criterion 1-5 (1=weak, 3=adequate, 5=exceptional).
5. Respond with ONLY valid JSON, no markdown fences, matching:
{"scores": {<criterion>: int, ...},
 "recommendation": "admit" | "borderline" | "reject",
 "justification": "<3-5 sentences>",
 "evidence_cited": ["E1", ...]}"""


def build_prompt(profile_min: dict, evidence: dict[str, list[dict]]) -> tuple[str, list[dict]]:
    flat, lines, eid = [], [], 0
    for crit, hits in evidence.items():
        for h in hits:
            eid += 1
            h = {**h, "eid": f"E{eid}"}
            flat.append(h)
            lines.append(f"[E{eid}] ({h['doc_type']}, criterion hint: {crit})\n{h['text']}")
    prompt = (
        f"Structured profile (pseudonymized):\n{json.dumps(profile_min, indent=2)}\n\n"
        f"Evidence chunks:\n\n" + "\n\n".join(lines) +
        f"\n\nEvaluate the applicant on: {', '.join(RUBRIC)}. Return JSON only."
    )
    return prompt, flat


def parse_response(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"error": "unparseable", "raw": raw}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"error": "invalid_json", "raw": raw}


def evaluate_applicant(applicant_id: str, profile_min: dict,
                       retriever, llm, audit_dir: str | Path) -> dict:
    evidence = retriever.retrieve_for_rubric(applicant_id)
    prompt, flat_evidence = build_prompt(profile_min, evidence)
    raw = llm.generate(SYSTEM_PROMPT, prompt)
    result = parse_response(raw)

    audit = {
        "applicant_id": applicant_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": getattr(llm, "model", "unknown"),
        "system_prompt": SYSTEM_PROMPT,
        "prompt": prompt,
        "evidence": flat_evidence,
        "raw_output": raw,
        "parsed_result": result,
        "disclaimer": "AI-assisted recommendation only. Final decision requires human review.",
    }
    d = Path(audit_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{applicant_id}_{int(time.time())}.json"
    path.write_text(json.dumps(audit, indent=2))
    return {"result": result, "evidence": flat_evidence, "audit_file": str(path)}
