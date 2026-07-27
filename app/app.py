"""Streamlit reviewer UI — transparent, human-in-the-loop evaluation.

Run: streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import yaml

CFG = yaml.safe_load(Path("config.yaml").read_text())

st.set_page_config(page_title="UniRAG Admissions Assistant", layout="wide")
st.title("UniRAG — Application Evaluation Assistant")
st.caption("AI-assisted recommendations. Final decisions require human review "
           "(EU AI Act: human oversight for high-risk systems).")


@st.cache_resource
def load_stack():
    from src.embeddings.embedder import Embedder
    from src.vectorstore.faiss_store import FaissStore
    from src.retrieval.retriever import Retriever
    e = CFG["embeddings"]
    embedder = Embedder(e["model_name"], e["batch_size"])
    store = FaissStore.load(CFG["data"]["index_dir"])
    return Retriever(store, embedder, top_k=CFG["retrieval"]["top_k"])


raw_dir = Path(CFG["data"]["raw_dir"])
applicants = sorted(d.name for d in raw_dir.iterdir() if d.is_dir())

tab_eval, tab_search, tab_audit = st.tabs(["Evaluate", "Semantic search", "Audit trail"])

with tab_eval:
    app_id = st.selectbox("Applicant", applicants)
    profile = json.loads((raw_dir / app_id / "profile.json").read_text())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GPA", f"{profile['gpa']}/4.0")
    c2.metric("IELTS", profile["ielts"])
    c3.metric("GRE Quant", profile["gre_quant"])
    c4.metric("Publications", profile["publications"])

    if st.button("Run evaluation", type="primary"):
        from src.privacy.anonymizer import minimize_profile
        from src.llm.local_llm import get_llm
        from src.orchestration.evaluator import evaluate_applicant
        with st.spinner("Retrieving evidence and querying local LLM..."):
            out = evaluate_applicant(app_id, minimize_profile(profile),
                                     load_stack(), get_llm(CFG["llm"]),
                                     CFG["data"]["audit_dir"])
        res = out["result"]
        if "scores" in res:
            st.subheader(f"Recommendation: **{res.get('recommendation', '?').upper()}**")
            st.bar_chart(res["scores"])
            st.write(res.get("justification", ""))
        else:
            st.error(res)
        with st.expander("Evidence used (full traceability)"):
            for e in out["evidence"]:
                st.markdown(f"**[{e['eid']}] {e['doc_type']}** (score {e['score']:.3f})")
                st.text(e["text"])
        st.caption(f"Audit trail written to `{out['audit_file']}`")

with tab_search:
    q = st.text_input("Query the whole applicant pool",
                      "strong research experience in machine learning")
    filt = st.selectbox("Filter to applicant", ["(all)"] + applicants)
    if st.button("Search"):
        hits = load_stack().retrieve(q, None if filt == "(all)" else filt)
        for h in hits:
            st.markdown(f"**{h['applicant_id']} / {h['doc_type']}** — score {h['score']:.3f}")
            st.text(h["text"][:400])

with tab_audit:
    audit_dir = Path(CFG["data"]["audit_dir"])
    files = sorted(audit_dir.glob("*.json"), reverse=True) if audit_dir.exists() else []
    if not files:
        st.info("No evaluations logged yet.")
    for f in files[:20]:
        with st.expander(f.name):
            st.json(json.loads(f.read_text()))
