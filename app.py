"""GroundLens: Pinecone semantic search with a LangGraph answering workflow."""
from __future__ import annotations

import os
from pathlib import Path
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from evaluation.benchmark import evaluate_retrieval, load_benchmark
from rag.pinecone_store import PineconeVectorStore
from rag.workflow import build_workflow

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
MANIFEST_PATH = ROOT / "runtime" / "pinecone_manifest.json"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "groundlens-rag-384").strip()
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "groundlens-v3").strip()
BENCHMARK_PATHS = {
    "Four academic PDFs": ROOT / "evaluation" / "academic_benchmark.csv",
    "Optional fictional-event samples": ROOT / "evaluation" / "benchmark_questions.csv",
}
MODEL_DEFAULT = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

st.set_page_config(page_title="GroundLens | Document Intelligence", page_icon="🔎",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root {
    --gl-burgundy: #5A1024;
    --gl-red: #C4123F;
    --gl-white: #FFFFFF;
    --gl-black: #000000;
    --gl-border: #C4123F;
}
.stApp, [data-testid="stAppViewContainer"] {
    background: #FFFFFF !important;
    color: var(--gl-black) !important;
}
[data-testid="stHeader"] { background: #FFFFFF !important; }
.block-container { max-width: 1240px; padding-top: 1.5rem; padding-bottom: 2rem; }
#MainMenu, footer { visibility: hidden; }

[data-testid="stSidebar"] {
    background: var(--gl-burgundy) !important;
    border-right: 2px solid var(--gl-red);
}
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(255,255,255,.08) !important;
    border: 1px solid rgba(255,255,255,.22) !important;
    border-radius: 12px;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section * { color: #FFFFFF !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18) !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--gl-red) !important;
    border: 1px solid var(--gl-red) !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #A80E35 !important;
    border-color: #A80E35 !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .st-emotion-cache-16txtl3,
[data-testid="stSidebar"] .st-emotion-cache-1r6slb0 {
    color: #FFFFFF !important;
    opacity: .95;
}

[data-testid="stMarkdownContainer"] .gl-hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, var(--gl-burgundy) 0%, var(--gl-red) 100%);
    border: 2px solid var(--gl-red);
    border-radius: 18px;
    padding: 1.55rem 1.9rem 1.45rem;
    margin: .15rem 0 1.35rem;
}
[data-testid="stMarkdownContainer"] .gl-hero::after {
    content: "";
    position: absolute;
    width: 240px; height: 240px;
    top: -140px; right: -55px;
    border-radius: 50%;
    background: rgba(255,255,255,.08);
}
[data-testid="stMarkdownContainer"] .gl-eyebrow {
    display: inline-block;
    padding: .35rem .72rem;
    border: 1px solid rgba(255,255,255,.35);
    border-radius: 999px;
    background: rgba(255,255,255,.08);
    color: #FFFFFF !important;
    font-size: .72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .09em;
}
[data-testid="stMarkdownContainer"] .gl-title {
    font-size: clamp(2.15rem, 3.8vw, 3.2rem);
    line-height: 1.08;
    font-weight: 820;
    margin: .8rem 0 .25rem !important;
    letter-spacing: -.045em;
    color: #FFFFFF !important;
}
[data-testid="stMarkdownContainer"] .gl-title span { color: #FFFFFF !important; }
[data-testid="stMarkdownContainer"] .gl-subtitle {
    color: #FFFFFF !important;
    font-size: 1rem;
    margin: .55rem 0 .72rem;
    line-height: 1.6;
}
.gl-heading-rule {
    width: 58px; height: 4px; border-radius: 6px;
    margin-top: 1rem; background: #FFFFFF;
}

[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 2px solid var(--gl-red);
    border-radius: 14px;
    padding: .85rem 1rem;
    min-height: 120px;
    box-shadow: none !important;
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #000000 !important; }

[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    border: 1.5px solid var(--gl-red);
    border-left: 4px solid var(--gl-burgundy);
    border-radius: 13px;
    margin: .65rem 0;
    box-shadow: none !important;
}
[data-testid="stChatMessage"] * { color: #000000 !important; }
[data-testid="stChatInput"] {
    border-radius: 12px;
    background: #FFFFFF !important;
}
textarea, input, [data-baseweb="textarea"], [data-baseweb="input"] {
    color: #000000 !important;
    background: #FFFFFF !important;
}
.stButton > button:not([kind="primary"]) {
    background: #FFFFFF !important;
    color: #000000 !important;
    border: 1.5px solid var(--gl-red) !important;
    border-radius: 10px;
}
.stButton > button:not([kind="primary"]):hover {
    color: #000000 !important;
    border-color: var(--gl-burgundy) !important;
}
button[data-baseweb="tab"] { font-weight: 650; color: #000000 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--gl-red) !important; }
[data-baseweb="tab-highlight"] { background: var(--gl-red) !important; }
[data-testid="stExpander"] {
    border: 1px solid var(--gl-red);
    border-radius: 11px;
    background: #FFFFFF !important;
}
[data-testid="stCode"] { border: 1px solid #D9D9D9; }
[data-testid="stAlertContainer"] * { color: #000000 !important; }
@media (max-width: 700px) {
    [data-testid="stMarkdownContainer"] .gl-hero { padding: 1.1rem 1.15rem; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Connecting to Pinecone vector database...")
def get_store():
    return PineconeVectorStore(
        api_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX,
        namespace=PINECONE_NAMESPACE,
        manifest_file=MANIFEST_PATH,
        cloud=os.getenv("PINECONE_CLOUD", "aws"),
        region=os.getenv("PINECONE_REGION", "us-east-1"),
    )


@st.cache_resource(show_spinner="Loading CPU embedding model (first run may download it)...")
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def sources_box(hits: list, expanded: bool = False) -> None:
    with st.expander(f"Evidence explorer · {len(hits)} retrieved passages", expanded=expanded):
        st.caption("Source IDs apply only to this answer. Pinecone cosine similarity "
                   "is a retrieval score, NOT a probability that an answer is correct.")
        for i, hit in enumerate(hits, 1):
            st.markdown(f"**[S{i}] {hit.filename} — page {hit.page}** · cosine similarity {1.0 - hit.distance:.3f}")
            st.code(hit.text, language=None, wrap_lines=True)


def safe_error(exc: Exception) -> str:
    message = str(exc)
    for key in (API_KEY, PINECONE_API_KEY):
        if key:
            message = message.replace(key, "[REDACTED]")
    return f"{type(exc).__name__}: {message[:700]}"


if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "benchmark_result" not in st.session_state:
    st.session_state.benchmark_result = None

try:
    store = get_store()
except Exception as exc:
    st.error("Pinecone setup is incomplete or unavailable: " + safe_error(exc))
    st.info("Create a Pinecone API key, fill the PINECONE_* values in your private `.env`, "
            "and restart. See README.md for first-time setup.")
    st.stop()
docs = store.list_documents()
doc_by_id = {d["doc_id"]: d for d in docs}

st.markdown("""<section class="gl-hero" aria-label="GroundLens document intelligence">
<div class="gl-eyebrow">PINECONE + LANGGRAPH · RAG MINI-PROJECT</div>
<h1 class="gl-title">Ground<span>Lens</span></h1>
<p class="gl-subtitle">Ask across PDFs with Pinecone retrieval, LangGraph workflows and page-level evidence.</p>
<div class="gl-heading-rule"></div>
</section>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📁 Source library")
    uploads = st.file_uploader("Upload PDF documents", type=["pdf"],
                               accept_multiple_files=True,
                               help="Maximum 12 MB per PDF. For scanned slides, use an OCR-searchable copy.")
    if st.button("Index selected PDFs", type="primary", use_container_width=True,
                 disabled=not uploads):
        try:
            model = get_embedder()
            indexed_any = False
            for upload in uploads or []:
                if upload.size > 12 * 1024 * 1024:
                    st.error(f"{upload.name}: exceeds 12 MB.")
                    continue
                try:
                    with st.spinner(f"Extracting and indexing {upload.name}..."):
                        status, chunks, pages = store.add_pdf(upload.name, upload.getvalue(), model)
                    if status == "already_indexed":
                        st.info(f"{upload.name}: identical PDF already indexed.")
                    else:
                        indexed_any = True
                        st.success(f"{upload.name}: {pages} pages and {chunks} chunks uploaded to Pinecone.")
                        st.caption("New Pinecone upserts can take a few seconds to become searchable.")
                except Exception as exc:
                    st.error(f"{upload.name}: {safe_error(exc)}")
            if indexed_any:
                st.session_state.conversation = []  # No stale source references.
                st.session_state.benchmark_result = None
                docs = store.list_documents()
                doc_by_id = {d["doc_id"]: d for d in docs}
        except Exception as exc:
            st.error("Cannot load local embedding model: " + safe_error(exc))

    st.caption(f"**{len(docs)} PDFs** · **{store.count} Pinecone chunks**")
    st.caption(f"Index: `{PINECONE_INDEX}` · namespace: `{PINECONE_NAMESPACE}`")
    st.caption("Uploaded text and vectors are stored in Pinecone's cloud; "
               "do not index confidential or redistribution-restricted documents.")
    st.divider()
    st.header("⚙️ Question settings")
    scope_ids = st.multiselect("Search within documents", options=list(doc_by_id),
                               default=list(doc_by_id),
                               format_func=lambda d: f"{doc_by_id[d]['filename']} · {d[:7]}",
                               help="Choose one or more PDFs; leave all selected to search the whole library.")
    top_k = st.slider("Evidence passages", min_value=2, max_value=8, value=5)
    balanced = st.checkbox("Cover multiple documents", value=False,
                           help="Reserve evidence from distinct documents when comparing files. "
                                "A LangGraph expansion step may also search individual comparison topics.")
    show_evidence = st.toggle("Expand evidence by default", value=False)
    model_name = st.text_input("Gemini model", value=MODEL_DEFAULT)
    if not API_KEY or API_KEY.startswith("REPLACE_"):
        st.warning("Gemini key missing. Create `.env` from `.env.example`.")
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.conversation = []
        st.rerun()

summary1, summary2, summary3 = st.columns(3)
summary1.metric("Indexed documents", len(docs))
summary2.metric("Pinecone vector chunks", store.count)
summary3.metric("LLM model", "Gemini" if model_name.lower().startswith("gemini") else (model_name or "Not configured"),
                help=f"Configured model: {model_name}")

ask_tab, library_tab, eval_tab, about_tab = st.tabs([
    "💬 Ask documents", "📚 Manage PDFs", "📊 Evaluate retrieval", "ℹ️ About"
])

with ask_tab:
    if not docs:
        st.info("Upload your permitted PDFs in the sidebar and click **Index selected PDFs**.")
    elif not scope_ids:
        st.warning("Select at least one document in the sidebar.")

    for previous in st.session_state.conversation:
        with st.chat_message("user"):
            st.write(previous["question"])
        with st.chat_message("assistant"):
            st.markdown(previous["answer"])
            st.caption(f"Retrieval {previous['retrieval_seconds']:.2f}s · "
                       f"Gemini {previous['generation_seconds']:.2f}s · "
                       f"Total {previous['total_seconds']:.2f}s")
            if previous["invalid_citations"]:
                st.warning(f"Unrecognized citation IDs: {previous['invalid_citations']}. "
                           "Check the source passages manually.")
            if previous.get("graph_steps"):
                st.caption("LangGraph path: " + " → ".join(previous["graph_steps"]))
            sources_box(previous["hits"])

    question = st.chat_input("Ask a question about the selected PDFs...",
                             disabled=not docs or not scope_ids)
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            try:
                started = time.perf_counter()
                with st.spinner("Running Pinecone retrieval and LangGraph workflow..."):
                    graph = build_workflow(
                        store, get_embedder(), gemini_api_key=API_KEY,
                        model_name=model_name.strip(),
                    )
                    result = graph.invoke({
                        "question": question, "doc_ids": scope_ids,
                        "top_k": top_k, "balanced": balanced,
                        "history": st.session_state.conversation[-2:],
                    })
                total_seconds = time.perf_counter() - started
                answer = result["answer"]
                hits = result.get("hits", [])
                retrieval_seconds = result.get("retrieval_seconds", 0.0)
                generation_seconds = result.get("generation_seconds", 0.0)
                invalid = result.get("invalid_citations", [])
                st.markdown(answer)
                st.caption(f"Retrieval {retrieval_seconds:.2f}s · "
                           f"Gemini {generation_seconds:.2f}s · Total {total_seconds:.2f}s")
                st.caption("LangGraph path: " + " → ".join(result.get("graph_steps", [])))
                if invalid:
                    st.warning(f"Unrecognized citation IDs: {invalid}. "
                               "Citations are labels, not automated fact verification.")
                sources_box(hits, expanded=show_evidence)
                st.session_state.conversation.append({
                    "question": question, "answer": answer, "hits": hits,
                    "retrieval_seconds": retrieval_seconds,
                    "generation_seconds": generation_seconds,
                    "total_seconds": total_seconds, "invalid_citations": invalid,
                    "graph_steps": result.get("graph_steps", []),
                })
            except Exception as exc:
                st.error("Unable to answer. " + safe_error(exc))
                st.caption("Check Pinecone connectivity, Gemini availability and source selection.")

with library_tab:
    st.subheader("Indexed document management")
    if docs:
        st.dataframe(pd.DataFrame([{
            "Document": d["filename"], "Pages": d["page_count"],
            "Indexed passages": d["chunks"], "Document ID": d["doc_id"][:12],
        } for d in docs]), hide_index=True, use_container_width=True)
        selected_delete = st.selectbox("Document to remove", options=list(doc_by_id),
                                       format_func=lambda d: f"{doc_by_id[d]['filename']} · {d[:8]}")
        confirm_delete = st.checkbox("I understand this deletes this document's vectors and text from Pinecone")
        if st.button("Delete selected document", disabled=not confirm_delete):
            store.delete_document(selected_delete)
            st.session_state.conversation = []
            st.session_state.benchmark_result = None
            st.rerun()
        with st.expander("Clear the GroundLens Pinecone namespace"):
            st.warning(f"Deletes ALL text and vectors in namespace `{PINECONE_NAMESPACE}` "
                       "only. The Pinecone index itself and other namespaces remain intact.")
            confirmation = st.checkbox("Confirm deletion of GroundLens namespace data")
            if st.button("Delete ALL indexed PDFs", disabled=not confirmation):
                store.clear_all()
                st.session_state.conversation = []
                st.session_state.benchmark_result = None
                st.rerun()
    else:
        st.info("No PDFs indexed yet. Add a searchable PDF using the sidebar.")
    st.caption("Duplicate uploads are recognized locally using SHA-256. "
               "Original PDFs remain on your computer; Pinecone stores searchable text and vectors. "
               "Document listings use a private local manifest.")

with eval_tab:
    st.subheader("Reproducible Pinecone retrieval benchmark")
    st.write("The benchmark tests whether required PDF passages appear in Pinecone search "
             "results. It makes Pinecone requests but **does not call Gemini**.")
    st.caption("Retrieval hit rate measures retrieved evidence, not generated-answer accuracy. "
               "Each benchmark searches only its own indexed source PDFs; missing sources are skipped.")
    benchmark_choice = st.selectbox("Question set", options=list(BENCHMARK_PATHS))
    benchmark_path = BENCHMARK_PATHS[benchmark_choice]
    if not benchmark_path.exists():
        st.error("Evaluation dataset is missing from the project folder.")
    else:
        benchmark = load_benchmark(benchmark_path)
        st.caption(f"{len(benchmark)} benchmark questions (including manual/unsupported cases).")
        bench_col1, bench_col2 = st.columns(2)
        with bench_col1:
            eval_k = st.select_slider("Evaluate top-k passages", options=[2, 3, 4, 5, 6, 8], value=5)
        with bench_col2:
            eval_balanced = st.checkbox("Balanced multi-document retrieval", value=True)
        if st.button("Run local benchmark", type="primary", disabled=not docs):
            try:
                with st.spinner("Retrieving evidence for benchmark questions..."):
                    active_names = {name.strip() for row in benchmark
                                    for name in row.get("expected_documents", "").split(";")
                                    if name.strip()}
                    active_ids = [d["doc_id"] for d in docs if d["filename"] in active_names]
                    result_rows, metrics = evaluate_retrieval(store, get_embedder(),
                                                              benchmark, k=eval_k,
                                                              balanced=eval_balanced,
                                                              allowed_doc_ids=active_ids)
                st.session_state.benchmark_result = (result_rows, metrics, eval_k, eval_balanced, benchmark_choice)
            except Exception as exc:
                st.error("Benchmark failed: " + safe_error(exc))
        if st.session_state.benchmark_result is not None:
            result_rows, metrics, evaluated_k, evaluated_balanced, dataset_name = st.session_state.benchmark_result
            if dataset_name != benchmark_choice:
                st.info("Results displayed below are from the previously selected question set. "
                        "Click Run local benchmark to evaluate the current selection.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Scored questions", metrics["evaluated"])
            c2.metric(f"Retrieval hit@{evaluated_k}",
                      f"{metrics['retrieval_hit_rate']:.1%}" if metrics["retrieval_hit_rate"] is not None else "N/A")
            c3.metric("MRR (all required sources)",
                      f"{metrics['mean_reciprocal_rank']:.3f}" if metrics["mean_reciprocal_rank"] is not None else "N/A")
            st.caption(f"{metrics['skipped_or_manual']} questions skipped/manual. "
                       f"Mode: {'balanced' if evaluated_balanced else 'standard'}. "
                       "The table displays measured results from this machine.")
            results_df = pd.DataFrame(result_rows)
            st.dataframe(results_df, hide_index=True, use_container_width=True)
            st.download_button("Download actual benchmark results (CSV)",
                               data=results_df.to_csv(index=False).encode("utf-8"),
                               file_name="groundlens_retrieval_results.csv", mime="text/csv")
        st.divider()
        st.markdown("**What still needs manual evaluation?**")
        st.write("Answer correctness, whether a citation really supports each claim, "
                 "and behavior on questions absent from documents. The included "
                 "`evaluation/manual_answer_tests.csv` provides a small checklist.")

with about_tab:
    st.subheader("How GroundLens works")
    st.code("PDF → PyMuPDF page text → 170-word chunks → local MiniLM vectors\n"
            "    → Pinecone serverless namespace (remote text + vectors)\n"
            "Question → LangGraph prepare → Pinecone retrieve\n"
            "    → [optional expand per subtopic] → [Gemini generate / abstain]\n"
            "    → LangGraph citation audit → answer + page references", language="text")
    st.markdown("**Course alignment:** Module 1 — PDF RAG + embeddings; "
                "Module 2 — Pinecone dense vector indexing and semantic search; "
                "Module 3 — actual LangGraph StateGraph with conditional paths; "
                "Module 4 — source inspection, latency timings and retrieval benchmark.")
    st.info("The workflow is implemented in `rag/workflow.py` using LangGraph nodes and "
            "conditional edges. This project does **not** use LangSmith tracing or "
            "Pinecone's integrated embedding service.")
    st.warning("Pinecone stores uploaded text and vector embeddings in the cloud. "
               "Gemini receives selected passages and questions. "
               "Use only documents permitted for these services. Scanned PDFs require OCR.")


