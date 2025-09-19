import os
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from helpers.graph_manager import GraphManager
from helpers import utils as U
from helpers.utils import (
    list_file_paths, load_path_as_documents, load_all_documents, _format_required
)

graph = GraphManager()

app = FastAPI(title="RAG + Internet Manager Graph API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class QueryRequest(BaseModel):
    thread_id: str = Field(..., description="Conversation/thread key for short-term memory")
    query: str = Field(..., description="User query")

class FinalResponse(BaseModel):
    manager_agent: Dict[str, Any]
    agent_responses: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=FinalResponse)
def query(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Send a query")
    final = graph.invoke(
        thread_id=req.thread_id if hasattr(req, "thread_id") else "default",
        query=req.query
    )
    try:
        return _format_required(final or {})
    except Exception:
        if isinstance(final, dict) and "manager_agent" in final and "agent_responses" in final:
            return {"manager_agent": final["manager_agent"], "agent_responses": final["agent_responses"]}
        return final or {}

@app.post("/embeddings/upload")
async def upload_and_build_embeddings(files: List[UploadFile] = File(...)):
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    os.makedirs(U.DATA_DIR, exist_ok=True)

    saved: List[str] = []
    skipped: List[str] = []

    for f in files:
        name = os.path.basename(f.filename or "").strip()
        if not name:
            skipped.append("(unnamed)")
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in U.SUPPORTED_EXTS:
            skipped.append(name)
            continue

        dest = os.path.join(U.DATA_DIR, name)
        content = await f.read()
        with open(dest, "wb") as out:
            out.write(content)
        saved.append(dest)

    if not saved:
        raise HTTPException(status_code=400, detail="No supported files were uploaded.")

    per_file_docs = {}
    for p in saved:
        try:
            per_file_docs[os.path.basename(p)] = len(load_path_as_documents(p))
        except Exception:
            per_file_docs[os.path.basename(p)] = 0

    current_files = [os.path.basename(p) for p in list_file_paths(U.DATA_DIR)]

    docs_count = U.rebuild_faiss_from_all_documents()

    return {
        "status": "ok",
        "saved_files": [os.path.basename(p) for p in saved],
        "skipped_files": skipped,
        "per_file_loaded_docs": per_file_docs,
        "total_supported_files_in_data": len(current_files),
        "documents_indexed": docs_count,
        "faiss_index_path": U.FAISS_DIR,
    }

@app.delete("/embeddings/reset")
def delete_files_and_index():
    """Delete all supported files in ./data and the entire FAISS index folder."""
    result = U.delete_data_and_index()
    return {"status": "ok", **result}

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=True)
