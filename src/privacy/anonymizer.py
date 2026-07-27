"""GDPR-oriented pseudonymization.

Before any text is embedded or sent to the LLM, we replace direct identifiers
(name, email) with the stable applicant ID. The mapping ID -> identity is kept
ONLY in profile.json on disk (the "controller's" key), never in the index or
prompts — classic pseudonymization under GDPR Art. 4(5).

For production, swap the regex layer for Microsoft Presidio or spaCy NER.
"""
from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def pseudonymize(text: str, profile: dict) -> str:
    """Replace known direct identifiers from the profile, plus generic PII patterns."""
    out = text
    name = profile.get("name", "")
    if name:
        out = out.replace(name, profile["applicant_id"])
        # also catch "First" / "Last" alone
        for part in name.split():
            if len(part) > 2:
                out = re.sub(rf"\b{re.escape(part)}\b", profile["applicant_id"], out)
    out = EMAIL_RE.sub("[EMAIL_REDACTED]", out)
    out = PHONE_RE.sub("[PHONE_REDACTED]", out)
    return out


def minimize_profile(profile: dict) -> dict:
    """Data minimization: keep only evaluation-relevant fields for prompting."""
    keep = ["applicant_id", "program", "intended_specialization", "undergrad_degree",
            "cgpa", "cgpa_scale", "german_equivalent_grade", "english_test",
            "english_score", "gre_quant", "work_experience_years", "internships",
            "publications", "interests", "aps_required",
            # legacy v1 fields, kept for backward compatibility
            "gpa", "gpa_scale", "ielts"]
    return {k: profile[k] for k in keep if k in profile}