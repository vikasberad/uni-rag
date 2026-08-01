"""Tests for the MCP server tool layer.

Only tools that need no embedding model or LLM are exercised directly; the
retrieval-backed tools are covered by monkeypatching a stub retriever, so the
suite stays fast and model-free.
"""
import json

import pytest

from src.mcp_server import server as mcp_server


class StubRetriever:
    def retrieve(self, query, applicant_id=None):
        aid = applicant_id or "APP-0001"
        return [{"chunk_id": f"{aid}:sop:{j}", "applicant_id": aid,
                 "doc_type": "sop", "text": f"evidence {j}", "score": 0.7}
                for j in range(6)]

    def retrieve_for_rubric(self, applicant_id, per_criterion=3):
        return {"academic_readiness": [
            {"chunk_id": f"{applicant_id}:transcript:0",
             "applicant_id": applicant_id, "doc_type": "transcript",
             "text": "CGPA 9.22", "score": 0.8}]}


@pytest.fixture
def stub_retriever(monkeypatch):
    monkeypatch.setattr(mcp_server, "get_retriever", lambda: StubRetriever())


def test_list_applicants_returns_pool():
    data = json.loads(mcp_server.list_applicants())
    assert data["count"] >= 1
    first = data["applicants"][0]
    assert first["applicant_id"].startswith("APP-")
    assert "program" in first and "german_equivalent_grade" in first


def test_profile_tool_strips_pii():
    """Privacy regression: identifiers must never cross the MCP boundary."""
    app_id = json.loads(mcp_server.list_applicants())["applicants"][0]["applicant_id"]
    profile = json.loads(mcp_server.get_applicant_profile(app_id))
    assert "name" not in profile
    assert "email" not in profile
    assert "date_of_birth" not in profile
    assert "cgpa" in profile


def test_unknown_applicant_returns_error():
    assert "error" in json.loads(mcp_server.get_applicant_profile("APP-9999"))
    assert "error" in json.loads(mcp_server.get_evidence_for_criteria("APP-9999"))


def test_search_respects_top_k(stub_retriever):
    data = json.loads(mcp_server.search_applications("research", top_k=3))
    assert len(data["results"]) == 3
    assert data["scope"] == "whole pool"


def test_compare_requires_two_ids(stub_retriever):
    ids = [a["applicant_id"] for a in
           json.loads(mcp_server.list_applicants())["applicants"][:2]]
    assert "error" in json.loads(mcp_server.compare_applicants(ids[0]))
    data = json.loads(mcp_server.compare_applicants(",".join(ids)))
    assert set(data["applicants"]) == set(ids)
    for entry in data["applicants"].values():
        assert "name" not in entry["profile"]
        assert len(entry["evidence"]) == 4


def test_evidence_by_criterion_shape(stub_retriever):
    app_id = json.loads(mcp_server.list_applicants())["applicants"][0]["applicant_id"]
    data = json.loads(mcp_server.get_evidence_for_criteria(app_id))
    assert "academic_readiness" in data["evidence_by_criterion"]
