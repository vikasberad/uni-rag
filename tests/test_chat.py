"""Tests for the professor chat engine (no models required — uses stubs)."""
import json
from pathlib import Path

from src.orchestration.chat import resolve_scope, build_chat_prompt, chat_turn

ALL_IDS = [f"APP-{i:04d}" for i in range(1, 26)]


class StubRetriever:
    def retrieve(self, q, applicant_id=None):
        aid = applicant_id or "APP-0001"
        return [{"chunk_id": f"{aid}:sop:{j}", "applicant_id": aid, "doc_type": "sop",
                 "text": f"evidence {j}", "score": 0.9} for j in range(6)]


class StubLLM:
    model = "stub"

    def generate(self, system, prompt):
        return "Grounded answer citing [E1]."


def test_resolve_scope_from_question_text():
    assert resolve_scope("Compare app-0002 and APP-0008", [], ALL_IDS) == \
        ["APP-0002", "APP-0008"]
    assert resolve_scope("unknown APP-9999", [], ALL_IDS) == []


def test_resolve_scope_merges_ui_selection():
    scope = resolve_scope("does APP-0005 have publications?", ["APP-0002"], ALL_IDS)
    assert scope == ["APP-0002", "APP-0005"]


def test_chat_turn_caps_evidence_and_writes_audit(tmp_path):
    out = chat_turn("Compare APP-0002 and APP-0008", [], ALL_IDS,
                    StubRetriever(), StubLLM(), "data/raw", tmp_path,
                    minimize=lambda p: {"applicant_id": p["applicant_id"]},
                    history=[{"role": "user", "content": "earlier question"}])
    assert out["scope"] == ["APP-0002", "APP-0008"]
    assert len(out["evidence"]) == 8  # 4 per applicant
    assert {e["applicant_id"] for e in out["evidence"]} == {"APP-0002", "APP-0008"}
    audit = json.loads(Path(out["audit_file"]).read_text())
    assert audit["type"] == "chat_turn" and audit["answer"]


def test_prompt_contains_history_profiles_and_evidence():
    prompt = build_chat_prompt(
        "who is stronger?",
        {"APP-0001": {"cgpa": 9.0}},
        [{"eid": "E1", "applicant_id": "APP-0001", "doc_type": "lor_1", "text": "praise"}],
        [{"role": "user", "content": "previous"}])
    assert "Conversation so far" in prompt
    assert "APP-0001" in prompt and "[E1]" in prompt
    assert "who is stronger?" in prompt
