"""Smoke tests for the non-model parts of the pipeline (run: pytest -q)."""
import json
from pathlib import Path

from src.ingestion.loader import load_applicant
from src.ingestion.chunker import chunk_documents
from src.privacy.anonymizer import pseudonymize, minimize_profile

RAW = Path("data/raw")


def _first_app():
    return sorted(d for d in RAW.iterdir() if d.is_dir())[0]


def test_loader():
    profile, docs = load_applicant(_first_app())
    assert profile["applicant_id"].startswith("APP-")
    assert {d.doc_type for d in docs} == {"sop", "lor_1", "lor_2", "cv", "transcript"}


def test_chunker():
    _, docs = load_applicant(_first_app())
    chunks = chunk_documents(docs, 300, 50)
    assert all(len(c.text) <= 360 for c in chunks)
    assert all(c.applicant_id for c in chunks)


def test_pseudonymize():
    profile, docs = load_applicant(_first_app())
    text = docs[0].text + f"\nContact: {profile['email']}"
    clean = pseudonymize(text, profile)
    assert profile["name"] not in clean
    assert profile["email"] not in clean


def test_minimize_profile():
    profile = json.loads((_first_app() / "profile.json").read_text())
    m = minimize_profile(profile)
    assert "name" not in m and "email" not in m and "date_of_birth" not in m
    assert "cgpa" in m and "german_equivalent_grade" in m
