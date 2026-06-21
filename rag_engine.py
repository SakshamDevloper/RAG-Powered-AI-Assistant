import os
import pickle
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "NousResearch/Hermes-3-Llama-3.1-8B"
VECTOR_STORE_DIR = Path("vectorstore")
VECTOR_STORE_PATH = VECTOR_STORE_DIR / "faiss_index"
DOCUMENTS_DIR = Path("documents")

VECTOR_STORE_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True)


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_llm(api_token: str, model_name: Optional[str] = None):
    return ChatOpenAI(
        model=model_name or LLM_MODEL,
        api_key=api_token,
        base_url="https://router.huggingface.co/v1",
        temperature=0.3,
        max_tokens=1024,
        top_p=0.95,
    )


def load_document(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def create_vector_store(documents: List[Document]) -> FAISS:
    embeddings = get_embeddings()
    return FAISS.from_documents(documents, embeddings)


def save_vector_store(vector_store: FAISS):
    vector_store.save_local(str(VECTOR_STORE_PATH))
    metadata = {
        "doc_count": vector_store.index.ntotal,
    }
    with open(VECTOR_STORE_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


def load_vector_store() -> Optional[FAISS]:
    if not VECTOR_STORE_PATH.exists():
        return None
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_vector_store() -> FAISS:
    vector_store = load_vector_store()
    if vector_store is None:
        vector_store = FAISS.from_documents(
            [Document(page_content="Placeholder. Ingest documents to begin.")],
            get_embeddings(),
        )
        save_vector_store(vector_store)
    return vector_store


def process_uploaded_file(uploaded_file) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        docs = load_document(tmp_path)
    finally:
        os.unlink(tmp_path)
    return docs


def ingest_documents(uploaded_files) -> int:
    all_docs = []
    for uploaded_file in uploaded_files:
        docs = process_uploaded_file(uploaded_file)
        all_docs.extend(docs)
    if not all_docs:
        return 0
    chunks = split_documents(all_docs)
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    save_vector_store(vector_store)
    return len(chunks)


class ReRankRetriever(BaseRetriever):
    """Retriever that fetches extra docs then re-ranks with a cross-encoder."""

    vector_store: FAISS
    search_type: str = "similarity"
    base_k: int = 20
    rerank_k: int = 4
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        base = self.vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs={"k": self.base_k},
        )
        docs = base._get_relevant_documents(query, run_manager=run_manager)
        if not docs:
            return docs

        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(self.reranker_model)
            pairs = [[query, d.page_content] for d in docs]
            scores = reranker.predict(pairs)
            scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            docs = [d for d, _ in scored[:self.rerank_k]]
        except Exception:
            docs = docs[:self.rerank_k]

        return docs


def extract_sources(documents: List[Document]) -> List[dict]:
    """Extract readable source metadata from retrieved documents."""
    seen = set()
    sources = []
    for doc in documents:
        meta = doc.metadata
        source = meta.get("source", "Unknown")
        page = meta.get("page", None)
        key = f"{source}:p{page}" if page is not None else source
        if key in seen:
            continue
        seen.add(key)
        label = Path(source).name if source != "Unknown" else "Unknown"
        entry = {"source": label, "page": page, "snippet": doc.page_content[:180].strip()}
        sources.append(entry)
    return sources


def get_retrieval_chain(
    huggingface_api_token: str,
    model_name: Optional[str] = None,
    search_type: str = "similarity",
    k: int = 4,
    rerank: bool = False,
    rerank_k: int = 4,
):
    llm = get_llm(huggingface_api_token, model_name)
    vector_store = get_vector_store()

    if rerank:
        retriever = ReRankRetriever(
            vector_store=vector_store,
            search_type=search_type,
            base_k=k * 3,
            rerank_k=rerank_k,
        )
    else:
        search_kwargs = {"k": k}
        if search_type == "mmr":
            search_kwargs["fetch_k"] = k * 2
            search_kwargs["lambda_mult"] = 0.7
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain


def ingest_documents_from_paths(file_paths: List[str]) -> int:
    all_docs = []
    for path in file_paths:
        docs = load_document(path)
        all_docs.extend(docs)
    if not all_docs:
        return 0
    chunks = split_documents(all_docs)
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    save_vector_store(vector_store)
    return len(chunks)


def clear_vector_store():
    if VECTOR_STORE_PATH.exists():
        import shutil
        shutil.rmtree(VECTOR_STORE_DIR)
        VECTOR_STORE_DIR.mkdir(exist_ok=True)
