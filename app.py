import os
import re
import streamlit as st
from pypdf import PdfReader
from transformers import pipeline

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ==================== PAGE ====================
st.set_page_config(page_title="AI Research Copilot | GPT + RAG", layout="wide")
st.title("🔬 AI Research Copilot")
st.caption("GPT-powered Research Copilot that analyzes academic papers using Retrieval-Augmented Generation (FAISS + semantic search). Provides cited answers, evidence extraction, and instant summaries.")


# ==================== SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vs" not in st.session_state:
    st.session_state.vs = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "paper_text" not in st.session_state:
    st.session_state.paper_text = ""


# ==================== MODELS ====================
@st.cache_resource
def get_qa_model():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")


def free_answer(question: str, context: str):
    """Returns (answer, score). Always returns 2 values."""
    qa = get_qa_model()
    result = qa(question=question, context=context[:6000])

    answer = (result.get("answer") or "").strip()
    score = float(result.get("score") or 0.0)

    if not answer or score < 0.35:
        return (
            "I couldn't find a clear answer in the retrieved context. Try asking about conclusion, methodology, or applications.",
            0.0,
        )

    return answer, score


# ==================== OPENAI (OPTIONAL) ====================
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if USE_OPENAI:
    from openai import OpenAI
    client = OpenAI()


def openai_answer_stream(question: str, context: str):
    """Streaming answer for chat (OpenAI)."""
    prompt = f"""You are an expert academic research assistant.
Respond in a clear, structured, university-level writing style.

Answer using ONLY the provided context.

VERY IMPORTANT:
- Cite sources like [S1], [S2]
- Do NOT make up information
- If unsure, say: "I couldn't find that in the paper."

Context:
{context}

Question: {question}

Answer:
"""

    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        stream=True,
    )


def openai_answer_once(prompt: str) -> str:
    """Non-stream call (useful for summary)."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ==================== HELPERS ====================
def read_pdf_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def get_index_folder(file_name: str) -> str:
    # Keep only ASCII letters/numbers/_/-
    safe = file_name.encode("ascii", "ignore").decode()
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", safe).strip("_")

    if not safe:
        safe = "paper_index"

    return os.path.join("faiss_indexes", safe)



def ensure_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text(text)

    # remove noisy chunks (links, too short)
    chunks = [
        c for c in chunks
        if len(c.strip()) > 80 and "http" not in c.lower() and "www." not in c.lower()
    ]
    return chunks


def build_or_load_vectorstore(text: str, file_name: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_folder = get_index_folder(file_name)

    # Load if exists
    if os.path.exists(index_folder):
        vs = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)
        return vs, None  # chunks will be regenerated from text (fast)

    # Build
    chunks = ensure_chunks(text)
    vs = FAISS.from_texts(chunks, embeddings)

    os.makedirs(index_folder, exist_ok=True)
    vs.save_local(index_folder)
    
    return vs, chunks


# ==================== UI ====================
pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if not pdf:
    st.info("Upload a PDF to start chatting.")
    st.stop()

with st.spinner("Reading PDF..."):
    text = read_pdf_text(pdf)

if not text.strip():
    st.error("I couldn't extract text from this PDF. If it's scanned (image-based), you need OCR.")
    st.stop()

st.success("PDF text extracted ✅")
st.session_state.paper_text = text

# Build/load FAISS index once per app session
if st.session_state.vs is None:
    with st.spinner("Preparing vector index (FAISS)..."):
        file_name = pdf.name.replace(".pdf", "")
        vs, chunks = build_or_load_vectorstore(text, file_name)
        st.session_state.vs = vs

        # If loaded from disk, regenerate chunks for summary button
        if chunks is None:
            chunks = ensure_chunks(text)

        st.session_state.chunks = chunks

    st.success("Vector index ready ✅")

# Chat header + clear button
colA, colB = st.columns([3, 1])
with colA:
    st.subheader("💬 Chat")
with colB:
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Show previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])


# -------- Summary Button --------
if st.button("📄 Summarize Paper"):
    chunks_for_summary = st.session_state.chunks[:8]
    summary_context = "\n\n".join(chunks_for_summary)

    if USE_OPENAI:
        try:
            summary_prompt = f"""Summarize this research paper in 6–10 bullet points.
Include: purpose, main topics, key points, applications, and conclusion.
Use ONLY the context below.

Context:
{summary_context}
"""
            summary = openai_answer_once(summary_prompt)
        except Exception:
            summary, _ = free_answer(
                "Summarize this paper in 6 bullet points: purpose, topics, applications, and conclusion.",
                summary_context,
            )
    else:
        summary, _ = free_answer(
            "Summarize this paper in 6 bullet points: purpose, topics, applications, and conclusion.",
            summary_context,
        )

    with st.chat_message("assistant"):
        st.write("### 📄 Paper Summary")
        st.write(summary)

    st.session_state.messages.append({"role": "assistant", "content": f"📄 Paper Summary:\n{summary}"})


# Quick question buttons
st.write("Quick questions:")
c1, c2, c3, c4 = st.columns(4)
quick_q = None
if c1.button("Main topic"):
    quick_q = "What is the main topic of this paper?"
if c2.button("Methodology"):
    quick_q = "What methodology or approach is described?"
if c3.button("Applications"):
    quick_q = "What applications are discussed in the paper?"
if c4.button("Conclusion"):
    quick_q = "What is the conclusion of the paper?"

# Chat input
question = st.chat_input("Ask anything about this research paper...")
if quick_q:
    question = quick_q


# ==================== RAG PIPELINE ====================
if question:
    # show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Retrieve sources
    with st.spinner("Searching paper..."):
        docs = st.session_state.vs.max_marginal_relevance_search(
            question, k=8, fetch_k=20
        )

    # Build labeled context for citations
    context = "\n\n".join([f"[S{i+1}] {d.page_content}" for i, d in enumerate(docs)])

    # --- Generate answer ---
    if USE_OPENAI:
        try:
            with st.spinner("Generating answer (OpenAI)..."):
                stream = openai_answer_stream(question, context)

            full_answer = ""
            with st.chat_message("assistant"):
                placeholder = st.empty()
                for event in stream:
                    delta = event.choices[0].delta.content or ""
                    full_answer += delta
                    placeholder.write(full_answer)

            answer = full_answer
        except Exception as e:
            # fallback to free mode
            answer, score = free_answer(question, context)
            with st.chat_message("assistant"):
                st.write(answer)
                st.caption(f"Confidence Score: {score:.2f}")
                st.caption("⚠️ Tip: If answer looks incorrect, ask about methodology, applications, or conclusion.")
    else:
        answer, score = free_answer(question, context)
        with st.chat_message("assistant"):
            st.write(answer)
            st.caption(f"Confidence Score: {score:.2f}")
            st.caption("⚠️ Tip: If answer looks incorrect, ask about methodology, applications, or conclusion.")

    # Save assistant message once
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # --- Evidence panel (docs always exists here) ---
    st.subheader("📚 Key Evidence From Paper")
    for i, d in enumerate(docs, 1):
        snippet = d.page_content[:320].replace("\n", " ")
        st.info(f"**Source {i}** — {snippet}...")
