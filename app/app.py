"""Streamlit reviewer UI — chat, evaluation, search and audit trail.

Run: streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import yaml

CFG = yaml.safe_load(Path("config.yaml").read_text())
RAW_DIR = Path(CFG["data"]["raw_dir"])

st.set_page_config(page_title="UniRAG Admissions Assistant", layout="wide")
st.title("UniRAG — Application Evaluation Assistant")
st.caption("AI-assisted recommendations. Final decisions require human review "
           "(EU AI Act: human oversight for high-risk systems).")


@st.cache_resource
def load_retriever():
    from src.embeddings.embedder import Embedder
    from src.vectorstore.faiss_store import FaissStore
    from src.retrieval.retriever import Retriever
    e = CFG["embeddings"]
    embedder = Embedder(e["model_name"], e["batch_size"])
    store = FaissStore.load(CFG["data"]["index_dir"])
    return Retriever(store, embedder, top_k=CFG["retrieval"]["top_k"])


@st.cache_resource
def load_llm():
    from src.llm.local_llm import get_llm
    return get_llm(CFG["llm"])


def load_profile(app_id: str) -> dict:
    return json.loads((RAW_DIR / app_id / "profile.json").read_text())


applicants = sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir())

tab_chat, tab_eval, tab_search, tab_audit = st.tabs(
    ["💬 Chat", "Evaluate", "Semantic search", "Audit trail"])

# ---------------------------------------------------------------- Chat tab
with tab_chat:
    st.subheader("Chat with the applicant pool")
    left, right = st.columns([3, 1])

    with right:
        scope_ids = st.multiselect(
            "Focus on applicant(s)", applicants,
            help="Leave empty to search the whole pool. Select 2+ to compare. "
                 "You can also just mention IDs like APP-0002 in your question.")
        show_profiles = st.checkbox("Show profile cards for focus", value=True)
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()
        if show_profiles:
            for a in scope_ids[:4]:
                p = load_profile(a)
                with st.container(border=True):
                    st.markdown(f"**{a}** — {p['program'].replace('MSc ', '')}")
                    st.caption(f"CGPA {p['cgpa']}/10 · DE {p['german_equivalent_grade']} · "
                               f"{p['english_test']} {p['english_score']} · "
                               f"{p['internships']} internship(s) · "
                               f"{p['publications']} publication(s)")

    with left:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("evidence"):
                    with st.expander(f"Evidence ({len(msg['evidence'])} chunks)"):
                        for e in msg["evidence"]:
                            st.markdown(f"**[{e['eid']}] {e['applicant_id']} / "
                                        f"{e['doc_type']}** (score {e['score']:.3f})")
                            st.text(e["text"][:500])

        question = st.chat_input(
            "e.g. 'Compare APP-0002 and APP-0008 for INFOTECH' or "
            "'Does APP-0013 have research experience?'")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving evidence and querying local LLM..."):
                    from src.privacy.anonymizer import minimize_profile
                    from src.orchestration.chat import chat_turn
                    out = chat_turn(
                        question=question,
                        selected_ids=scope_ids,
                        all_ids=applicants,
                        retriever=load_retriever(),
                        llm=load_llm(),
                        raw_dir=RAW_DIR,
                        audit_dir=CFG["data"]["audit_dir"],
                        minimize=minimize_profile,
                        history=st.session_state.chat_history[:-1],
                    )
                st.write(out["answer"])
                if out["scope"]:
                    st.caption("Scope: " + ", ".join(out["scope"]))
                if out["evidence"]:
                    with st.expander(f"Evidence ({len(out['evidence'])} chunks)"):
                        for e in out["evidence"]:
                            st.markdown(f"**[{e['eid']}] {e['applicant_id']} / "
                                        f"{e['doc_type']}** (score {e['score']:.3f})")
                            st.text(e["text"][:500])
                st.caption(f"Audit: `{out['audit_file']}`")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": out["answer"],
                 "evidence": out["evidence"]})

# ------------------------------------------------------------ Evaluate tab
with tab_eval:
    app_id = st.selectbox("Applicant", applicants)
    profile = load_profile(app_id)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CGPA", f"{profile['cgpa']}/10")
    c2.metric("German grade", profile["german_equivalent_grade"])
    c3.metric(profile["english_test"], profile["english_score"])
    c4.metric("Internships", profile["internships"])
    c5.metric("Publications", profile["publications"])
    st.caption(f"Program: {profile['program']} · Specialization: "
               f"{profile['intended_specialization']} · "
               f"Bachelor: {profile['undergrad_degree']}")

    if st.button("Run evaluation", type="primary"):
        from src.privacy.anonymizer import minimize_profile
        from src.orchestration.evaluator import evaluate_applicant
        with st.spinner("Retrieving evidence and querying local LLM..."):
            out = evaluate_applicant(app_id, minimize_profile(profile),
                                     load_retriever(), load_llm(),
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

# ------------------------------------------------------- Semantic search tab
with tab_search:
    q = st.text_input("Query the whole applicant pool",
                      "strong research experience in embedded systems")
    filt = st.selectbox("Filter to applicant", ["(all)"] + applicants)
    if st.button("Search"):
        hits = load_retriever().retrieve(q, None if filt == "(all)" else filt)
        for h in hits:
            st.markdown(f"**{h['applicant_id']} / {h['doc_type']}** — score {h['score']:.3f}")
            st.text(h["text"][:400])

# --------------------------------------------------------------- Audit tab
with tab_audit:
    audit_dir = Path(CFG["data"]["audit_dir"])
    files = sorted(audit_dir.glob("*.json"), reverse=True) if audit_dir.exists() else []
    if not files:
        st.info("No evaluations or chat turns logged yet.")
    for f in files[:25]:
        with st.expander(f.name):
            st.json(json.loads(f.read_text()))
