"""Load applicant documents from data/raw into structured Document objects."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DOC_TYPES = {"sop": "Statement of Purpose", "lor_1": "Recommendation Letter 1",
             "lor_2": "Recommendation Letter 2", "cv": "CV", "transcript": "Transcript"}


@dataclass
class Document:
    applicant_id: str
    doc_type: str          # sop | lor_1 | lor_2 | cv | transcript
    text: str
    metadata: dict = field(default_factory=dict)


def load_applicant(app_dir: Path) -> tuple[dict, list[Document]]:
    profile = json.loads((app_dir / "profile.json").read_text())
    docs = []
    for key, label in DOC_TYPES.items():
        f = app_dir / f"{key}.txt"
        if f.exists():
            docs.append(Document(
                applicant_id=profile["applicant_id"],
                doc_type=key,
                text=f.read_text(),
                metadata={"doc_label": label, "program": profile["program"]},
            ))
    return profile, docs


def load_all(raw_dir: str | Path) -> list[tuple[dict, list[Document]]]:
    raw = Path(raw_dir)
    return [load_applicant(d) for d in sorted(raw.iterdir()) if d.is_dir()]
