import os
import uuid
import uvicorn
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

from rag_engine import (
    ingest_documents,
    get_retrieval_chain,
    clear_vector_store,
    extract_sources,
)

app = FastAPI(title="Aura RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversations: dict[str, list[dict]] = {}

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    hf_token: str | None = None
    model_name: str | None = None
    retrieval_k: int = 4
    search_type: str = "similarity"
    rerank_enabled: bool = False
    rerank_k: int = 4

class ChatResponse(BaseModel):
    reply: str
    sources: list[dict]
    suggestions: list[str]
    conversation_id: str

@app.get("/")
async def serve_index():
    idx = Path(__file__).parent / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"error": "index.html not found"}, status_code=404)

@app.post("/api/chat")
async def chat(req: ChatRequest):
    token = req.hf_token or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    if not token:
        raise HTTPException(status_code=400, detail="HF token required")

    cid = req.conversation_id or str(uuid.uuid4())
    if cid not in conversations:
        conversations[cid] = []

    chain = get_retrieval_chain(
        token,
        model_name=req.model_name,
        search_type=req.search_type,
        k=req.retrieval_k,
        rerank=req.rerank_enabled,
        rerank_k=req.rerank_k,
    )

    result = chain.invoke({"question": req.message})
    reply = result.get("answer", "I couldn't find an answer.")
    reply = __import__("re").sub(r'^.*?Answer:\s*', '', reply, flags=__import__("re").IGNORECASE | __import__("re").DOTALL).strip()
    source_docs = result.get("source_documents", [])
    sources = extract_sources(source_docs)

    conversations[cid].append({"role": "user", "content": req.message})
    conversations[cid].append({"role": "assistant", "content": reply, "sources": sources})

    suggestions = [
        "Summarize the key points from the documents",
        "What are the main topics covered?",
        "Explain this in more detail",
        "List important findings and conclusions",
    ]

    return ChatResponse(reply=reply, sources=sources, suggestions=suggestions, conversation_id=cid)

@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    saved = []
    for f in files:
        dest = Path("documents") / f.filename
        content = await f.read()
        dest.write_bytes(content)
        saved.append(f.filename)
    count = ingest_documents([f for f in files])
    return {"files": saved, "chunks": count}

@app.get("/api/conversations")
async def list_conversations():
    return [
        {"id": cid, "messages": msgs}
        for cid, msgs in conversations.items()
    ]

@app.delete("/api/conversations/{cid}")
async def delete_conversation(cid: str):
    conversations.pop(cid, None)
    return {"status": "deleted"}

@app.get("/api/status")
async def status():
    from rag_engine import load_vector_store
    vs = load_vector_store()
    doc_count = vs.index.ntotal if vs is not None else 0
    return {"chunks": doc_count, "conversations": len(conversations)}

@app.post("/api/clear")
async def clear():
    clear_vector_store()
    return {"status": "cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
