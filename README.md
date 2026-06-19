# Aura RAG — Intelligent AI Assistant

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** based AI Assistant that combines the power of large language models with dynamic knowledge retrieval. Instead of relying solely on pre-trained knowledge, this assistant can access and process external documents to provide more accurate, contextual, and up-to-date responses.

### Why RAG?
- **Contextual Awareness**: Ground responses in actual documents
- **Reduced Hallucinations**: Minimize fabricated information
- **Dynamic Knowledge**: Update knowledge base without retraining
- **Transparency**: Always cite sources from your documents
- **Scalability**: Efficiently handle large document collections

---

## Features

- **Intelligent Document Processing** — Automatically ingest and process PDF, DOCX, TXT, CSV
- **Smart Retrieval** — Semantically search through your knowledge base with FAISS vector search
- **Context-Aware Responses** — Generate answers grounded in your documents via LLM
- **Source Citation** — See exactly which document and page the answer came from
- **Optional Re-ranking** — Cross-encoder re-ranker improves result quality
- **Conversation Memory** — Remembers previous messages for follow-up questions
- **Streamlit UI** — Clean, interactive web interface with chat, knowledge base, and settings

---

## Architecture

```
Documents → Chunking → Embeddings → Vector Database (FAISS)
                                          ↓
User Query → Embedding → Retriever → Relevant Chunks
                                          ↓
                              LLM (HuggingFace Router)
                                          ↓
                              Final Answer + Source Citations
```

### Pipeline Modules
1. **Data Collection** — Upload PDFs, DOCX, TXT, CSV
2. **Document Chunking** — Split into 1000-char chunks with overlap
3. **Embedding Generation** — `all-MiniLM-L6-v2` (384-dim vectors)
4. **Vector Database** — FAISS for fast similarity search
5. **Retrieval Engine** — Cosine similarity or MMR search + optional re-ranking
6. **LLM Generation** — HuggingFace OpenAI-compatible router (supports 200+ models)

---

## Quick Start

### Prerequisites
- Python 3.8+
- HuggingFace API token ([get one free](https://huggingface.co/settings/tokens))

### Installation

```bash
git clone https://github.com/SakshamDevloper/RAG-Powered-AI-Assistant.git
cd RAG-Powered-AI-Assistant
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your HuggingFace token:
# HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

### Run

```bash
python -m streamlit run app.py
```

### Ingest Documents

Via the Web UI — go to **Knowledge Base** tab and upload files.

Or via CLI:
```bash
python ingest.py path/to/document.pdf
python ingest.py path/to/directory --recursive
```

---

## Usage

1. Open `http://localhost:8501` in your browser
2. Go to **Settings** → enter your HuggingFace token
3. Go to **Knowledge Base** → upload documents
4. Go to **Chat** → ask questions about your documents

### Optional Features
- **Re-ranking**: Toggle on in Settings for higher accuracy (retrieves 3x chunks then re-ranks with cross-encoder)
- **Search Strategy**: Choose `similarity` (cosine) or `mmr` (diversity-aware)
- **LLM Model**: Change in Settings to any model from the HuggingFace router (e.g., `NousResearch/Hermes-3-Llama-3.1-8B`, `moonshotai/Kimi-K2-Instruct-0905`)

---

## Project Structure

```
RAG-Powered-AI-Assistant/
├── app.py              # Streamlit web UI (chat, knowledge base, settings)
├── rag_engine.py       # Core RAG logic (chunking, embeddings, FAISS, retrieval, LLM)
├── ingest.py           # CLI tool for batch document ingestion
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
├── README.md           # This file
└── documents/          # (optional) Place documents here for CLI ingestion
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | HuggingFace Router (OpenAI-compatible) |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Vector DB** | FAISS (local) |
| **Re-ranker** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| **Framework** | LangChain + Streamlit |
| **Language** | Python 3.13 |

---

## Troubleshooting

### Blank page in browser
Hard refresh with `Ctrl+Shift+R` or open in an incognito/private window.

### Model not supported
The default model may not be available on the HF router. Go to **Settings** and try:
- `NousResearch/Hermes-3-Llama-3.1-8B`
- `moonshotai/Kimi-K2-Instruct-0905`
- `Qwen/Qwen2.5-7B-Instruct`

### No documents found
Upload documents via the **Knowledge Base** tab, or use `python ingest.py <file>`.

---

## License

MIT License — see [LICENSE](LICENSE) file for details.

---

## Author

**Saksham Developer**
- GitHub: [@SakshamDevloper](https://github.com/SakshamDevloper)

---

<div align="center">
Made with Python
</div>
