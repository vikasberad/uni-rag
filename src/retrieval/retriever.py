"""Semantic retriever: embeds the query and searches FAISS with applicant filtering."""
from __future__ import annotations

from src.embeddings.embedder import Embedder
from src.vectorstore.faiss_store import FaissStore

# Rubric-aligned probe queries: for each evaluation criterion we retrieve the
# most relevant evidence chunks across the applicant's documents.
CRITERION_QUERIES = {
    "academic_readiness": "academic performance, grades, GPA, coursework, transcript strength",
    "research_potential": "research experience, thesis, publications, methodology, projects",
    "motivation_fit": "motivation, goals, why this program, alignment with program strengths",
    "recommendations": "recommender assessment, strengths, weaknesses, ranking among students",
    "experience_skills": "work experience, technical skills, software projects, internships",
}


class Retriever:
    def __init__(self, store: FaissStore, embedder: Embedder, top_k: int = 6):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str, applicant_id: str | None = None) -> list[dict]:
        qv = self.embedder.embed_query(query)
        return self.store.search(qv, top_k=self.top_k, applicant_id=applicant_id)

    def retrieve_for_rubric(self, applicant_id: str, per_criterion: int = 3) -> dict[str, list[dict]]:
        """Retrieve evidence per rubric criterion, deduplicated by chunk_id."""
        evidence, seen = {}, set()
        for crit, q in CRITERION_QUERIES.items():
            qv = self.embedder.embed_query(q)
            hits = self.store.search(qv, top_k=per_criterion, applicant_id=applicant_id)
            fresh = [h for h in hits if h["chunk_id"] not in seen]
            seen.update(h["chunk_id"] for h in fresh)
            evidence[crit] = fresh
        return evidence
