"""GroundLens v2 final: auditable multi-PDF question answering and retrieval evaluation."""
from __future__ import annotations

import io
import os
from pathlib import Path
import re
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from evaluation.benchmark import evaluate_retrieval, load_benchmark
from rag.generation import answer_with_gemini, invalid_source_numbers, INSUFFICIENT
from rag.vector_store import LocalVectorStore

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
INDEX_PATH = ROOT / "runtime" / "chroma"
BENCHMARK_PATHS = {
    "Four academic subjects": ROOT / "evaluation" / "four_subject_benchmark.csv",
    "Fictional event (original demo)": ROOT / "evaluation" / "benchmark_questions.csv",
}
MODEL_DEFAULT = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

st.set_page_config(page_title="GroundLens | Document Intelligence", page_icon="🔎",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
/* GroundLens NMIMS-inspired palette: accessible red, ivory and rose. */
:root {
    --gl-red: #B70F2E;
    --gl-red-dark: #8D1028;
    --gl-ink: #302629;
    --gl-ivory: #FAF7F2;
    --gl-sidebar: #F4EDE6;
    --gl-border: #E7DAD5;
}
.stApp, [data-testid="stAppViewContainer"] {
    background: linear-gradient(157deg, #FDFBF8 0%, #FAF7F2 52%, #F7F1EC 100%) !important;
    color: var(--gl-ink);
}
[data-testid="stHeader"] { background: rgba(253,251,248,.97) !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F6EFE9 0%, #F2E8E1 100%) !important;
    border-right: 1px solid #E7DAD5;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--gl-ink) !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #695857; }
.block-container { max-width: 1240px; padding-top: 1.75rem; padding-bottom: 2.25rem; }
#MainMenu, footer { visibility: hidden; }

/* Hero is explicitly contrasted: unlike a faint eyebrow on ivory. */
[data-testid="stMarkdownContainer"] .gl-hero {
    position: relative;
    isolation: isolate;
    overflow: hidden;
    background: linear-gradient(115deg, #FFFDFB 0%, #FCEFEB 56%, #F8E5E4 100%);
    border: 1px solid #EFDAD7;
    border-radius: 19px;
    padding: 1.45rem 1.85rem 1.4rem;
    margin: .25rem 0 1.4rem;
    box-shadow: 0 8px 26px rgba(115,45,50,.045);
}
[data-testid="stMarkdownContainer"] .gl-hero::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 5px;
    background: #B70F2E;
}
[data-testid="stMarkdownContainer"] .gl-hero::after {
    content: "";
    position: absolute;
    width: 245px; height: 245px;
    top: -160px; right: -65px;
    border-radius: 50%;
    background: rgba(183,15,46,.055);
    pointer-events: none;
}
[data-testid="stMarkdownContainer"] .gl-eyebrow {
    display: inline-block;
    padding: .35rem .7rem;
    border: 1px solid #EABFC5;
    border-radius: 99px;
    background: #F9E3E6;
    color: #8D1028 !important;
    opacity: 1 !important;
    font-size: .72rem;
    font-weight: 800;
    line-height: 1.25;
    text-transform: uppercase;
    letter-spacing: .09em;
}
[data-testid="stMarkdownContainer"] .gl-title {
    font-size: clamp(2.15rem, 3.8vw, 3.2rem);
    line-height: 1.09;
    font-weight: 820;
    margin: .8rem 0 .2rem !important;
    letter-spacing: -.045em;
    color: #302629 !important;
}
[data-testid="stMarkdownContainer"] .gl-title span { color: #B70F2E !important; }
[data-testid="stMarkdownContainer"] .gl-subtitle {
    color: #655354 !important;
    font-size: .97rem;
    margin: .55rem 0 .7rem;
    line-height: 1.6;
}
.gl-heading-rule {
    width: 58px;
    height: 4px;
    background: #B70F2E;
    border-radius: 6px;
    margin-top: 1rem;
}

/* Information cards and answers have crisp white surfaces. */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E7DAD5;
    border-top: 3px solid #B70F2E;
    border-radius: 14px;
    padding: .85rem 1rem;
    min-height: 120px;
    box-shadow: 0 5px 19px rgba(95,46,44,.055);
}
[data-testid="stMetricLabel"] { color: #6E5756 !important; }
[data-testid="stMetricValue"] { color: #302629 !important; }
[data-testid="stChatMessage"] {
    background: #FFFFFF;
    border: 1px solid #E7DAD5;
    border-left: 3px solid #B70F2E;
    border-radius: 13px;
    margin: .65rem 0;
    box-shadow: 0 3px 13px rgba(95,46,44,.04);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { line-height: 1.62; }
[data-testid="stFileUploader"] { border-color: #E7DAD5; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
    background: #B70F2E !important;
    border: 1px solid #B70F2E !important;
    color: #FFFFFF !important;
    border-radius: 10px;
    font-weight: 650;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
    background: #8D1028 !important;
    border-color: #8D1028 !important;
}
.stButton > button:not([kind="primary"]) {
    border-color: #DCC8C3;
    border-radius: 10px;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #B70F2E;
    color: #B70F2E;
}
button[data-baseweb="tab"] { font-weight: 650; }
button[data-baseweb="tab"][aria-selected="true"] { color: #B70F2E !important; }
[data-baseweb="tab-highlight"] { background: #B70F2E !important; }
[data-testid="stExpander"] { border-color: #E7DAD5; border-radius: 11px; background: #FFF; }
[data-testid="stChatInput"] { border-radius: 12px; background-color: #FFFDFB; }
[data-testid="stCode"] { border: 1px solid #E9DED9; }
@media (max-width: 700px) {
    [data-testid="stMarkdownContainer"] .gl-hero { padding: 1.1rem 1.15rem; }
    [data-testid="stMarkdownContainer"] .gl-eyebrow { font-size: .62rem; letter-spacing: .065em; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Connecting to local document index...")
def get_store():
    return LocalVectorStore(INDEX_PATH)


@st.cache_resource(show_spinner="Loading CPU embedding model (first run may download it)...")
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def sources_box(hits: list, expanded: bool = False) -> None:
    with st.expander(f"Evidence explorer · {len(hits)} retrieved passages", expanded=expanded):
        st.caption("IDs [S1] etc refer to these passages for this specific answer. "
                   "Vector distance is NOT an answer-confidence score.")
        for i, hit in enumerate(hits, 1):
            st.markdown(f"**[S{i}] {hit.filename} — page {hit.page}** · distance {hit.distance:.3f}")
            st.code(hit.text, language=None, wrap_lines=True)


def safe_error(exc: Exception) -> str:
    message = str(exc)
    if API_KEY:
        message = message.replace(API_KEY, "[REDACTED]")
    return f"{type(exc).__name__}: {message[:700]}"


if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "benchmark_result" not in st.session_state:
    st.session_state.benchmark_result = None

store = get_store()
docs = store.list_documents()
doc_by_id = {d["doc_id"]: d for d in docs}

st.markdown("""<section class="gl-hero" aria-label="GroundLens document intelligence">
<div class="gl-eyebrow" style="color:#8D1028 !important; opacity:1 !important; background:#F9E3E6;">DOCUMENT INTELLIGENCE · RAG MINI-PROJECT</div>
<h1 class="gl-title" style="color:#302629 !important;">Ground<span style="color:#B70F2E !important;">Lens</span></h1>
<p class="gl-subtitle">Ask across your documents, with source-level evidence.</p>
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
                        st.success(f"{upload.name}: {pages} pages and {chunks} chunks indexed.")
                except Exception as exc:
                    st.error(f"{upload.name}: {safe_error(exc)}")
            if indexed_any:
                st.session_state.conversation = []  # No stale source references.
                st.session_state.benchmark_result = None
                docs = store.list_documents()
                doc_by_id = {d["doc_id"]: d for d in docs}
        except Exception as exc:
            st.error("Cannot load local embedding model: " + safe_error(exc))

    st.caption(f"**{len(docs)} PDFs** · **{store.count} local text chunks**")
    st.divider()
    st.header("⚙️ Question settings")
    scope_ids = st.multiselect("Search within documents", options=list(doc_by_id),
                               default=list(doc_by_id),
                               format_func=lambda d: f"{doc_by_id[d]['filename']} · {d[:7]}",
                               help="Choose one or more PDFs; leave all selected to search the whole library.")
    top_k = st.slider("Evidence passages", min_value=2, max_value=8, value=5)
    balanced = st.checkbox("Cover multiple documents", value=False,
                           help="Reserve one retrieved passage per document when comparing files. "
                                "Turn this off for ordinary single-fact queries.")
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
summary2.metric("Local evidence chunks", store.count)
summary3.metric("LLM model", "Gemini" if model_name.lower().startswith("gemini") else (model_name or "Not configured"),
                help=f"Configured model: {model_name}")

ask_tab, library_tab, eval_tab, about_tab = st.tabs([
    "💬 Ask documents", "📚 Manage PDFs", "📊 Evaluate retrieval", "ℹ️ About"
])

with ask_tab:
    if not docs:
        st.info("Upload a PDF in the sidebar and click **Index selected PDFs**. "
                "Three fictional PDFs are provided in `sample_documents/`.")
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
            sources_box(previous["hits"])

    question = st.chat_input("Ask a question about the selected PDFs...",
                             disabled=not docs or not scope_ids)
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            try:
                start = time.perf_counter()
                with st.spinner("Retrieving passages locally..."):
                    retrieval_query = question
                    # Only a lightweight reference-resolution heuristic, not a full agent.
                    if (st.session_state.conversation and len(question.split()) <= 12
                            and re.search(r"\b(it|they|them|that|those|their|he|she)\b|\bwhat about\b",
                                          question, flags=re.IGNORECASE)):
                        retrieval_query = st.session_state.conversation[-1]["question"] + " " + question
                    hits = store.search(retrieval_query, get_embedder(), n_results=top_k,
                                        doc_ids=scope_ids, balanced=balanced)
                retrieval_seconds = time.perf_counter() - start
                generate_start = time.perf_counter()
                if not hits:
                    answer = INSUFFICIENT
                else:
                    with st.spinner("Generating an answer from the retrieved evidence..."):
                        answer = answer_with_gemini(API_KEY, model_name.strip(), question,
                                                    hits, st.session_state.conversation)
                generation_seconds = time.perf_counter() - generate_start
                total_seconds = time.perf_counter() - start
                invalid = invalid_source_numbers(answer, len(hits))
                st.markdown(answer)
                st.caption(f"Retrieval {retrieval_seconds:.2f}s · "
                           f"Gemini {generation_seconds:.2f}s · Total {total_seconds:.2f}s")
                if invalid:
                    st.warning(f"Gemini referred to missing sources {invalid}. "
                               "Citation IDs were checked for existence, not factual support.")
                sources_box(hits, expanded=show_evidence)
                st.session_state.conversation.append({
                    "question": question, "answer": answer, "hits": hits,
                    "retrieval_seconds": retrieval_seconds,
                    "generation_seconds": generation_seconds,
                    "total_seconds": total_seconds, "invalid_citations": invalid,
                })
            except Exception as exc:
                st.error("Unable to answer. " + safe_error(exc))
                st.caption("Verify the Gemini model, API quota, connection and indexed PDFs.")

with library_tab:
    st.subheader("Indexed document management")
    if docs:
        st.dataframe(pd.DataFrame([{
            "Document": d["filename"], "Pages": d["page_count"],
            "Indexed passages": d["chunks"], "Document ID": d["doc_id"][:12],
        } for d in docs]), hide_index=True, use_container_width=True)
        selected_delete = st.selectbox("Document to remove", options=list(doc_by_id),
                                       format_func=lambda d: f"{doc_by_id[d]['filename']} · {d[:8]}")
        confirm_delete = st.checkbox("I understand this deletes this document's local index")
        if st.button("Delete selected document", disabled=not confirm_delete):
            store.delete_document(selected_delete)
            st.session_state.conversation = []
            st.session_state.benchmark_result = None
            st.rerun()
        with st.expander("Clear entire local index"):
            st.warning("This permanently removes local indexed text, not your original PDFs.")
            confirmation = st.checkbox("Confirm deletion of all indexed documents")
            if st.button("Delete ALL indexed PDFs", disabled=not confirmation):
                store.clear_all()
                st.session_state.conversation = []
                st.session_state.benchmark_result = None
                st.rerun()
    else:
        st.info("No PDFs indexed yet. Add a searchable PDF using the sidebar.")
    st.caption("Duplicate uploads are recognized by SHA-256 file hash. "
               "The app does not retain original PDFs, only their indexed extracted passages.")

with eval_tab:
    st.subheader("Reproducible local retrieval benchmark")
    st.write("A small built-in benchmark tests whether the **correct source passage** appears "
             "among the retrieved results. It requires no Gemini API calls.")
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
    st.code("PDF upload → PyMuPDF text extraction → page-aware chunking\n"
            "→ local MiniLM embeddings → local ChromaDB index\n"
            "Question → local semantic retrieval → source-labelled passages\n"
            "→ Gemini text generation → answer + inspectable PDF/page sources", language="text")
    st.markdown("**Course alignment:** Module 1 — RAG pipeline and embeddings; "
                "Module 2 — local vector database and semantic retrieval; "
                "Module 3 — short conversational context and multi-step retrieval/generation; "
                "Module 4 — traceable evidence, separate latency measurements, and evaluation.")
    st.info("This project uses a lightweight Python workflow, not LangGraph or LangSmith. "
            "Do not claim those tools were implemented in the report.")
    st.warning("Use searchable PDFs (OCR first for scanned slides). Source IDs point to retrieved passages, "
               "not independently verified facts. Selected text and questions are sent to Gemini.")


