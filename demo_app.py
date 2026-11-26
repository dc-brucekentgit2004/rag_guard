"""Streamlit demo: shows each claim highlighted green / amber / red.

Run (with the API already running on port 8000):
    streamlit run demo_app.py
"""
import html

import httpx
import streamlit as st

API = st.sidebar.text_input("Guard API URL", "http://localhost:8000")
COLORS = {"supported": "#c8f7c5", "unverifiable": "#ffe9a8", "contradicted": "#ffb3b3"}

st.title("RAG Guard: hallucination check")
mode = st.radio("Mode", ["Ask my RAG assistant", "Paste an answer + sources"], horizontal=True)

question, answer, contexts = "", "", []
if mode == "Ask my RAG assistant":
    question = st.text_input("Question", "What is the punishment for cheating under Section 420 IPC?")
else:
    from rag_guard.doc_loader import read_document

    question = st.text_input("Question (optional)")
    answer = st.text_area("LLM answer (e.g. ChatGPT's answer)", height=150)

    st.markdown("**Source documents**")
    files = st.file_uploader("Upload the document(s) the answer should come from",
                             type=["pdf", "txt", "md", "docx"], accept_multiple_files=True)
    for f in files or []:
        try:
            text = read_document(f.name, f.getvalue())
            contexts.append(f"[{f.name}] {text}")
            with st.expander(f"{f.name}: {len(text.split()):,} words read (click to preview)"):
                st.write(text[:3000] + (" ..." if len(text) > 3000 else ""))
        except Exception as e:
            st.error(f"{f.name}: {e}")

    raw = st.text_area("Or paste source text (separate passages with a blank line)", height=120)
    contexts += [c.strip() for c in raw.split("\n\n") if c.strip()]


def highlight(text: str, claims: list) -> str:
    out, pos = [], 0
    for c in sorted(claims, key=lambda c: c["start"]):
        out.append(html.escape(text[pos:c["start"]]))
        tip = f'{c["label"]} (entail={c["entailment"]:.2f})'
        out.append(f'<span title="{tip}" style="background:{COLORS[c["label"]]};'
                   f'padding:2px 3px;border-radius:4px">{html.escape(text[c["start"]:c["end"]])}</span>')
        pos = c["end"]
    out.append(html.escape(text[pos:]))
    return "".join(out)


if st.button("Check", type="primary"):
    if mode != "Ask my RAG assistant":
        if not answer.strip():
            st.warning("Paste the LLM answer first.")
            st.stop()
        if not contexts:
            st.warning("Upload a document or paste source text first.")
            st.stop()
    with st.spinner("Checking claims against sources..."):
        if mode == "Ask my RAG assistant":
            r = httpx.post(f"{API}/ask", json={"question": question}, timeout=300)
        else:
            r = httpx.post(f"{API}/guard", json={"question": question, "answer": answer,
                                                 "contexts": contexts}, timeout=300)
    if r.status_code != 200:
        st.error(r.text)
        st.stop()
    rep = r.json()
    original = rep.get("original_answer", answer)

    c1, c2, c3 = st.columns(3)
    c1.metric("Decision", rep["decision"].upper())
    c2.metric("Faithfulness", f'{rep["faithfulness"]:.0%}')
    c3.metric("Unsupported claims", f'{rep["n_unsupported"]} / {rep["n_claims"]}')

    st.subheader("Original answer, claim by claim")
    st.markdown(highlight(original, rep["claims"]), unsafe_allow_html=True)
    st.caption("Green = supported, amber = not found in sources, red = contradicts sources")

    st.subheader("What the user sees")
    st.write(rep["final_answer"])

    with st.expander("Evidence for each claim"):
        for c in rep["claims"]:
            st.markdown(f'**{c["label"]}** ({c["entailment"]:.2f}): {c["text"]}')
            st.caption(c["evidence"][:400])