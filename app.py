import os
import re
import time
from pathlib import Path
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

from rag_engine import (
    ingest_documents,
    get_retrieval_chain,
    clear_vector_store,
    load_vector_store,
    extract_sources,
)

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

st.set_page_config(
    page_title="Aura RAG",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CINEMATIC DARK CSS — inspired by Aura landing page
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    html, body, [class*="css"] { font-family: system-ui, -apple-system, sans-serif; -webkit-font-smoothing: antialiased; }

    .stApp {
        background: #07080a;
        color: #fff;
    }

    ::selection { background: rgba(0,210,255,0.25); }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.06); border-radius: 2px; }

    section[data-testid="stSidebar"] { background: rgba(8,10,14,0.95) !important; border-right: 1px solid rgba(255,255,255,0.03) !important; }
    .sidebar-header { display: flex; align-items: center; gap: 12px; padding: 22px 16px 10px; }
    .sidebar-logo { width: 36px; height: 36px; flex-shrink: 0; filter: drop-shadow(0 0 14px rgba(0,210,255,0.2)); }
    .sidebar-title { font-size: 1.3rem; font-weight: 800; letter-spacing: -0.04em; background: linear-gradient(135deg, #A4F4FD, #00d2ff); -webkit-background-clip: text; background-clip: text; color: transparent; -webkit-text-fill-color: transparent; }
    .sidebar-sub { font-size: 0.6rem; color: rgba(255,255,255,0.18); letter-spacing: 0.4em; text-transform: uppercase; margin-top: -2px; }
    .stSidebar .stButton button { background: transparent !important; border: none !important; border-radius: 8px !important; color: rgba(255,255,255,0.4) !important; font-size: 0.8rem !important; font-weight: 500 !important; padding: 9px 14px !important; width: 100% !important; text-align: left !important; transition: all 0.2s ease !important; }
    .stSidebar .stButton button:hover { background: rgba(255,255,255,0.04) !important; color: #fff !important; }
    .sidebar-divider { height: 1px; background: rgba(255,255,255,0.03); margin: 6px 14px; }

    .main-chat { max-width: 760px; margin: 0 auto; padding: 16px 20px; }

    .chat-hero { text-align: center; padding: 70px 20px 36px; }
    .chat-hero h1 { font-size: 3rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1.05; margin: 0; }
    .chat-hero h1 span.gradient { background: linear-gradient(to right, #0B2551, #A4F4FD 30%, #00d2ff 50%, #A4F4FD 70%, #0B2551); background-size: 200% auto; -webkit-background-clip: text; background-clip: text; color: transparent; -webkit-text-fill-color: transparent; animation: shiny 4s linear infinite; }
    @keyframes shiny { 0% { background-position: -200% center; } 100% { background-position: 200% center; } }
    .chat-hero p { color: rgba(255,255,255,0.28); font-size: 0.86rem; margin-top: 12px; max-width: 360px; margin-left: auto; margin-right: auto; line-height: 1.65; }

    .chat-messages { display: flex; flex-direction: column; gap: 5px; padding-bottom: 10px; }
    .message-row { display: flex; align-items: flex-start; gap: 10px; animation: msgIn 0.3s ease; }
    .message-row.user { flex-direction: row-reverse; }
    @keyframes msgIn { 0% { opacity: 0; transform: translateY(6px); } 100% { opacity: 1; transform: translateY(0); } }
    .message-avatar { width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; font-weight: 600; margin-top: 2px; }
    .message-avatar.agent { background: linear-gradient(135deg, #00d2ff, #0066cc); color: #fff; box-shadow: 0 0 10px rgba(0,210,255,0.15); }
    .message-avatar.user-av { background: rgba(255,255,255,0.04); color: rgba(255,255,255,0.5); border: 1px solid rgba(255,255,255,0.06); }
    .message-bubble { max-width: 70%; padding: 9px 14px; border-radius: 12px; font-size: 0.86rem; line-height: 1.5; }
    .message-row.agent .message-bubble { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); color: rgba(255,255,255,0.85); border-bottom-left-radius: 2px; }
    .message-row.user .message-bubble { background: rgba(0,210,255,0.06); border: 1px solid rgba(0,210,255,0.08); color: #fff; border-bottom-right-radius: 2px; }

    .typing-indicator { display: flex; align-items: center; gap: 4px; padding: 8px 14px; }
    .typing-dot { width: 6px; height: 6px; border-radius: 50%; background: #00d2ff; animation: dotB 1.2s infinite; box-shadow: 0 0 4px rgba(0,210,255,0.2); }
    .typing-dot:nth-child(2) { animation-delay: 0.15s; }
    .typing-dot:nth-child(3) { animation-delay: 0.3s; }
    @keyframes dotB { 0%,60%,100% { opacity: 0.2; transform: translateY(0); } 30% { opacity: 1; transform: translateY(-3px); } }

    .input-container { max-width: 760px; margin: 0 auto; padding: 6px 0 20px; }
    .stTextInput > div { border: 1px solid rgba(255,255,255,0.05) !important; border-radius: 12px !important; background: rgba(255,255,255,0.015) !important; transition: all 0.2s ease !important; }
    .stTextInput > div:hover { border-color: rgba(255,255,255,0.1) !important; }
    .stTextInput > div:focus-within { border-color: rgba(0,210,255,0.2) !important; box-shadow: 0 0 0 3px rgba(0,210,255,0.04), 0 0 20px rgba(0,210,255,0.02) !important; }
    .stTextInput input { color: #fff !important; font-size: 0.86rem !important; padding: 10px 14px !important; }
    .stTextInput input::placeholder { color: rgba(255,255,255,0.16) !important; }

    .glass-card { background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.03); border-radius: 12px; position: relative; overflow: hidden; }
    .glass-card-content { padding: 16px; }

    .agent-status { display: flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 100px; background: rgba(0,210,255,0.04); border: 1px solid rgba(0,210,255,0.06); font-size: 0.65rem; color: rgba(255,255,255,0.4); white-space: nowrap; }
    .agent-status .pulse { width: 4px; height: 4px; border-radius: 50%; background: #00d2ff; animation: pulse 2s infinite; box-shadow: 0 0 4px rgba(0,210,255,0.3); }
    @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(0,210,255,0.3); } 50% { box-shadow: 0 0 0 5px rgba(0,210,255,0); } }

    .knowledge-stat { font-size: 0.66rem; color: rgba(255,255,255,0.22); line-height: 1.6; padding: 8px 16px; }
    .knowledge-stat strong { color: rgba(255,255,255,0.45); font-weight: 600; }

    .stFileUploader > div { border: 1px dashed rgba(255,255,255,0.06) !important; border-radius: 10px !important; background: rgba(255,255,255,0.01) !important; }
    .stFileUploader > div:hover { border-color: rgba(0,210,255,0.15) !important; }
    .stButton > button[kind="secondary"] { border-radius: 100px !important; border: 1px solid rgba(255,255,255,0.06) !important; background: transparent !important; color: rgba(255,255,255,0.4) !important; font-size: 0.76rem !important; padding: 3px 16px !important; transition: all 0.2s ease !important; }
    .stButton > button[kind="secondary"]:hover { border-color: rgba(255,255,255,0.12) !important; color: #fff !important; }

    .streamlit-expanderHeader { color: rgba(255,255,255,0.35) !important; font-size: 0.76rem !important; }
    .message-bubble p { margin: 0 0 3px; }
    .message-bubble p:last-child { margin-bottom: 0; }
    .message-bubble code { background: rgba(255,255,255,0.05); padding: 1px 4px; border-radius: 3px; font-size: 0.82em; }
    .message-bubble pre { background: rgba(0,0,0,0.3); border-radius: 7px; padding: 10px; overflow-x: auto; border: 1px solid rgba(255,255,255,0.03); }
    .message-bubble strong { color: #fff; }

    @media (max-width: 768px) { .chat-hero h1 { font-size: 1.6rem; } .message-bubble { max-width: 82%; font-size: 0.84rem; } }

    .stTabs [data-baseweb="tab-list"] { gap: 3px; background: rgba(255,255,255,0.015); border-radius: 8px; padding: 3px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px !important; color: rgba(255,255,255,0.35) !important; font-size: 0.76rem !important; font-weight: 500 !important; }
    .stTabs [aria-selected="true"] { background: rgba(255,255,255,0.05) !important; color: #fff !important; }

    [data-testid="stMetric"] { background: rgba(255,255,255,0.015); border: 1px solid rgba(255,255,255,0.03); border-radius: 10px; padding: 10px 14px; }
    [data-testid="stMetric"] label { color: rgba(255,255,255,0.3) !important; font-size: 0.68rem !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #fff !important; font-weight: 600 !important; }

    .stSpinner > div { border-color: rgba(0,210,255,0.3) rgba(0,210,255,0.1) rgba(0,210,255,0.1) rgba(0,210,255,0.3) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE INIT (must be before sidebar)
# ─────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "hf_token" not in st.session_state:
    st.session_state.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
if "model_name" not in st.session_state:
    st.session_state.model_name = os.getenv("MODEL_NAME", "NousResearch/Hermes-3-Llama-3.1-8B")
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "retrieval_k" not in st.session_state:
    st.session_state.retrieval_k = 4
if "retrieval_search_type" not in st.session_state:
    st.session_state.retrieval_search_type = "similarity"
if "rerank_enabled" not in st.session_state:
    st.session_state.rerank_enabled = False
if "rerank_k" not in st.session_state:
    st.session_state.rerank_k = 4

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR — macOS-style nav
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
            <svg class="sidebar-logo" viewBox="0 0 256 256" fill="none">
                <path d="M 0 128 C 70.692 128 128 185.308 128 256 L 64 256 C 64 220.654 35.346 192 0 192 Z M 256 192 C 220.654 192 192 220.654 192 256 L 128 256 C 128 185.308 185.308 128 256 128 Z M 128 0 C 128 70.692 70.692 128 0 128 L 0 64 C 35.346 64 64 35.346 64 0 Z M 192 0 C 192 35.346 220.654 64 256 64 L 256 128 C 185.308 128 128 70.692 128 0 Z" fill="white"/>
            </svg>
            <div>
                <div class="sidebar-title">Aura</div>
                <div class="sidebar-sub">Intelligent Agent</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Knowledge base stats
    @st.cache_data(ttl=5)
    def _get_chunk_count():
        vs = load_vector_store()
        return vs.index.ntotal if vs is not None else 0
    doc_count = _get_chunk_count()
    st.markdown(
        f"""
        <div class="knowledge-stats">
            <div class="knowledge-stat"><strong>{doc_count}</strong> chunks &mdash; <strong>FAISS</strong> index &mdash; <strong>{'Rerank ON' if st.session_state.rerank_enabled else 'Rerank OFF'}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Navigation ──
    if "nav" not in st.session_state:
        st.session_state.nav = "chat"

    nav_chat = st.button("Chat", key="nav_chat", use_container_width=True)
    nav_ingest = st.button("Knowledge Base", key="nav_ingest", use_container_width=True)
    nav_settings = st.button("Settings", key="nav_settings", use_container_width=True)

    if nav_chat:
        st.session_state.nav = "chat"
    elif nav_ingest:
        st.session_state.nav = "ingest"
    elif nav_settings:
        st.session_state.nav = "settings"

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Agent status indicator
    st.markdown(
        """
        <div class="agent-status">
            <div class="pulse"></div>
            Agent online &mdash; ready
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("")

# ─────────────────────────────────────────────────────────────────────
# RENDER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────
def render_chat():
    st.markdown('<div class="main-chat">', unsafe_allow_html=True)

    # ── Hero ──
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="chat-hero">
                <h1>Your knowledge.<br><span class="gradient">Intelligent.</span></h1>
                <p>Ask anything. Aura reads your documents and delivers answers with context, clarity, and speed.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Messages ──
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        ts = msg.get("time", "")
        is_user = role == "user"

        avatar_class = "user-av" if is_user else "agent"
        avatar_label = "U" if is_user else "A"
        row_class = "user" if is_user else "agent"

        st.markdown(
            f"""
            <div class="message-row {row_class}">
                <div class="message-avatar {avatar_class}">{avatar_label}</div>
                <div class="message-bubble"><div>{content}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show source citations for assistant messages
        if not is_user and msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    label = src["source"]
                    page = src.get("page")
                    snippet = src.get("snippet", "")
                    page_str = f" (p. {page})" if page is not None else ""
                    st.markdown(f"**{label}{page_str}**")
                    st.caption(f"> {snippet}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Typing indicator ──
    if st.session_state.get("thinking", False):
        st.markdown(
            """
            <div class="message-row agent">
                <div class="message-avatar agent">A</div>
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Suggestions (shown when no messages) ──
    if not st.session_state.messages:
        suggestions = [
            "Summarize my documents",
            "What are the key topics?",
            "Explain the main concepts",
            "List important findings",
        ]
        cols = st.columns(4)
        for i, s in enumerate(suggestions):
            if cols[i].button(s, key=f"sug_{i}", use_container_width=True):
                _handle_user_input(s)
                st.rerun()

    # ── Input ──
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    user_input = st.chat_input("Ask Aura anything...", key="chat_input")
    if user_input:
        _handle_user_input(user_input)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def _handle_user_input(text: str):
    st.session_state.messages.append({"role": "user", "content": text, "time": ""})
    st.session_state.thinking = True

    try:
        chain = _get_or_create_chain()
        if chain is None:
            reply = "Please configure your Hugging Face API token in Settings first."
            sources = []
        else:
            result = chain.invoke({"question": text})
            reply = result.get("answer", "I couldn't find an answer.")
            reply = re.sub(r'^.*?Answer:\s*', '', reply, flags=re.IGNORECASE | re.DOTALL).strip()
            if not reply:
                reply = result.get("answer", "I couldn't find an answer.").strip()
            source_docs = result.get("source_documents", [])
            sources = extract_sources(source_docs)
    except Exception as e:
        reply = f"Error: {e}"
        sources = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "time": "",
        "sources": sources,
    })
    st.session_state.thinking = False


def _get_or_create_chain():
    if st.session_state.chain is not None:
        return st.session_state.chain
    token = st.session_state.hf_token
    if not token:
        return None
    chain = get_retrieval_chain(
        token,
        st.session_state.model_name,
        search_type=st.session_state.retrieval_search_type,
        k=st.session_state.retrieval_k,
        rerank=st.session_state.rerank_enabled,
        rerank_k=st.session_state.rerank_k,
    )
    st.session_state.chain = chain
    return chain


def render_ingest():
    st.markdown('<div class="main-chat">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="chat-hero" style="padding-bottom: 16px;">
            <h1 style="font-size:2rem;">Knowledge <span class="gradient">Base</span></h1>
            <p>Upload PDFs, DOCX, CSV, or TXT files. Aura will read and index them for conversational retrieval.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Current stats
    vs = load_vector_store()
    doc_count = vs.index.ntotal if vs is not None else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Indexed Chunks", doc_count)
    c2.metric("Similarity Engine", "FAISS")
    c3.metric("Embedding Model", "all-MiniLM-L6-v2")

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload
    uploaded_files = st.file_uploader(
        "Drop documents here",
        type=["pdf", "txt", "docx", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        file_key = tuple(f.name for f in uploaded_files)
        if st.session_state.get('_last_ingested') != file_key:
            with st.spinner("Ingesting documents..."):
                count = ingest_documents(uploaded_files)
            vs = load_vector_store()
            new_count = vs.index.ntotal if vs is not None else 0
            st.success(f"Ingested {count} chunks. Total: {new_count}.")
            st.session_state.chain = None
            st.session_state._last_ingested = file_key
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Clear all documents", type="secondary"):
        clear_vector_store()
        st.session_state.chain = None
        st.session_state.messages = []
        st.session_state._last_ingested = None
        st.success("Vector store cleared.")
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── RAG Pipeline modules (for college project / PPT) ──
    st.markdown(
        """
        <div class="glass-card"><div class="glass-card-content">
        <span style="color:rgba(255,255,255,0.5); font-size:0.7rem; letter-spacing:0.15em; text-transform:uppercase;">RAG Pipeline</span>
        <div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:10px;">
            <span style="padding:4px 12px; border-radius:100px; border:1px solid rgba(0,210,255,0.2); background:rgba(0,210,255,0.06); color:rgba(255,255,255,0.7); font-size:0.72rem;">1. Data Collection</span>
            <span style="padding:4px 12px; border-radius:100px; border:1px solid rgba(0,210,255,0.2); background:rgba(0,210,255,0.06); color:rgba(255,255,255,0.7); font-size:0.72rem;">2. Chunking</span>
            <span style="padding:4px 12px; border-radius:100px; border:1px solid rgba(0,210,255,0.2); background:rgba(0,210,255,0.06); color:rgba(255,255,255,0.7); font-size:0.72rem;">3. Embeddings</span>
            <span style="padding:4px 12px; border-radius:100px; border:1px solid rgba(0,210,255,0.2); background:rgba(0,210,255,0.06); color:rgba(255,255,255,0.7); font-size:0.72rem;">4. Vector DB</span>
            <span style="padding:4px 12px; border-radius:100px; border:1px solid rgba(0,210,255,0.2); background:rgba(0,210,255,0.06); color:rgba(255,255,255,0.7); font-size:0.72rem;">5. Retrieval</span>
            <span style="padding:4px 12px; border-radius:100px; border:1px solid rgba(0,210,255,0.2); background:rgba(0,210,255,0.06); color:rgba(255,255,255,0.7); font-size:0.72rem;">6. LLM Generation</span>
            <span style="padding:4px 12px; border-radius:100px; border:1px solid rgba(0,210,255,0.2); background:rgba(0,210,255,0.06); color:rgba(255,255,255,0.7); font-size:0.72rem;">+ Source Citation</span>
            <span style="padding:4px 12px; border-radius:100px; border:1px solid rgba(255,200,0,0.2); background:rgba(255,200,0,0.06); color:rgba(255,255,255,0.7); font-size:0.72rem;">+ Re-ranking</span>
        </div>
        </div></div>
        """,
        unsafe_allow_html=True,
    )

    # Document list
    st.markdown(
        """
        <div style="margin-top: 24px;">
            <span class="sidebar-pill">supported formats</span>
            <span style="color:rgba(255,255,255,0.3); font-size:0.78rem; margin-left:8px;">
                PDF &middot; DOCX &middot; TXT &middot; CSV
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_settings():
    st.markdown('<div class="main-chat">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="chat-hero" style="padding-bottom: 16px;">
            <h1 style="font-size:2rem;"><span class="gradient">Settings</span></h1>
            <p>Configure your LLM backend and API credentials.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="glass-card"><div class="glass-card-content">', unsafe_allow_html=True)

        st.markdown("#### Hugging Face")
        token = st.text_input(
            "API Token",
            value=st.session_state.hf_token,
            type="password",
            placeholder="hf_...",
            help="Get your token at huggingface.co/settings/tokens",
        )
        if token != st.session_state.hf_token:
            st.session_state.hf_token = token
            st.session_state.chain = None

        model = st.text_input(
            "Model ID",
            value=st.session_state.model_name,
            placeholder="mistralai/Mistral-7B-Instruct-v0.3",
            help="Any Hugging Face text-generation model. Must be free / accessible with your token.",
        )
        if model != st.session_state.model_name:
            st.session_state.model_name = model
            st.session_state.chain = None

        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Retrieval config ──
    with st.container():
        st.markdown('<div class="glass-card"><div class="glass-card-content">', unsafe_allow_html=True)
        st.markdown("#### Retrieval")

        search_type = st.selectbox(
            "Search strategy",
            options=["similarity", "mmr"],
            index=0 if st.session_state.retrieval_search_type == "similarity" else 1,
            help="similarity = top-k by cosine distance · mmr = diversity-aware ranking",
        )
        if search_type != st.session_state.retrieval_search_type:
            st.session_state.retrieval_search_type = search_type
            st.session_state.chain = None

        k = st.slider(
            "Number of chunks to retrieve (k)",
            min_value=1, max_value=20, value=st.session_state.retrieval_k,
            help="Higher = more context but slower responses and higher token usage.",
        )
        if k != st.session_state.retrieval_k:
            st.session_state.retrieval_k = k
            st.session_state.chain = None

        st.markdown(
            f"""
            <div style="color:rgba(255,255,255,0.4); font-size:0.75rem; margin-top:8px;">
                Current: <strong>k={st.session_state.retrieval_k}</strong> · search: <strong>{st.session_state.retrieval_search_type}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="glass-card"><div class="glass-card-content">', unsafe_allow_html=True)
        st.markdown("#### Re-ranking")

        rerank_enabled = st.toggle(
            "Enable re-ranking (cross-encoder)",
            value=st.session_state.rerank_enabled,
            help="Retrieves more chunks then re-ranks them with a cross-encoder for higher relevance.",
        )
        if rerank_enabled != st.session_state.rerank_enabled:
            st.session_state.rerank_enabled = rerank_enabled
            st.session_state.chain = None

        if st.session_state.rerank_enabled:
            rerank_k = st.slider(
                "Top chunks after re-ranking",
                min_value=1, max_value=10, value=st.session_state.rerank_k,
                help="How many of the re-ranked chunks to keep.",
            )
            if rerank_k != st.session_state.rerank_k:
                st.session_state.rerank_k = rerank_k
                st.session_state.chain = None

        st.markdown(
            f"""
            <div style="color:rgba(255,255,255,0.4); font-size:0.75rem; margin-top:8px;">
                Re-ranking: <strong>{"ON" if st.session_state.rerank_enabled else "OFF"}</strong>
                {" · top " + str(st.session_state.rerank_k) if st.session_state.rerank_enabled else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model info
    with st.container():
        st.markdown('<div class="glass-card"><div class="glass-card-content">', unsafe_allow_html=True)
        st.markdown("#### About")
        st.markdown(
            """
            <div style="color:rgba(255,255,255,0.5); font-size:0.85rem; line-height:1.6;">
                <strong style="color:rgba(255,255,255,0.8);">Aura RAG Agent</strong> uses:
                <br> <strong>LangChain</strong> — conversational retrieval chain with memory
                <br> <strong>FAISS</strong> — vector similarity search
                <br> <strong>HuggingFace Router</strong> — OpenAI-compatible LLM endpoint
                <br> <strong>Cross-Encoder</strong> — optional re-ranking
                <br> <strong>Streamlit</strong> — interactive frontend
                <br><br>
                Embedding model: <code>all-MiniLM-L6-v2</code> (384 dims)
                <br>Default LLM: <code>Hermes-3-Llama-3.1-8B</code>
                <br>Re-ranker: <code>ms-marco-MiniLM-L-6-v2</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE ROUTING
# ─────────────────────────────────────────────────────────────────────
if st.session_state.nav == "chat":
    render_chat()
elif st.session_state.nav == "ingest":
    render_ingest()
elif st.session_state.nav == "settings":
    render_settings()
