import os
import re
import glob
import warnings
import shutil
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pypdf.errors import PdfReadWarning
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchResults

warnings.filterwarnings("ignore", category=PdfReadWarning)
warnings.filterwarnings(
    "ignore",
    message="This package (`duckduckgo_search`) has been renamed to `ddgs`!",
    category=RuntimeWarning,
    module="langchain_community.utilities.duckduckgo_search",
)

load_dotenv()
DATA_DIR = "./data"
FAISS_DIR = "./faiss_bge_index"

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

faiss_store: Optional[FAISS] = None

SUPPORTED_EXTS = [".pdf", ".txt", ".md", ".csv", ".tsv", ".xlsx", ".xls", ".docx", ".doc"]

def list_file_paths(folder: str = DATA_DIR, exts: Optional[List[str]] = None) -> List[str]:
    os.makedirs(folder, exist_ok=True)
    exts = exts or SUPPORTED_EXTS
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(paths)

def load_path_as_documents(path: str) -> List[Document]:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(path).load()

    if ext in {".txt", ".md"}:
        return TextLoader(path, encoding="utf-8").load()

    if ext in {".csv", ".tsv"}:
        csv_args = {"delimiter": "\t"} if ext == ".tsv" else None
        return CSVLoader(file_path=path, csv_args=csv_args).load()

    if ext in {".xlsx", ".xls"}:
        try:
            return UnstructuredExcelLoader(path).load()
        except Exception as e:
            print(f"[WARN] UnstructuredExcelLoader failed for {path}: {e}")
            try:
                import pandas as pd
                xls = pd.ExcelFile(path)
                parts = []
                for sheet in xls.sheet_names:
                    df = xls.parse(sheet)
                    parts.append(f"### Sheet: {sheet}\n" + df.to_csv(index=False))
                content = "\n\n".join(parts)
                return [Document(page_content=content, metadata={"source": path, "type": "excel_fallback"})]
            except Exception as e2:
                print(f"[WARN] Pandas fallback failed for {path}: {e2}")
                return []

    if ext == ".docx":
        try:
            return Docx2txtLoader(path).load()
        except Exception as e:
            print(f"[INFO] Docx2txtLoader failed for {path}: {e}; trying UnstructuredWordDocumentLoader")
            try:
                return UnstructuredWordDocumentLoader(path).load()
            except Exception as e2:
                print(f"[WARN] UnstructuredWordDocumentLoader failed for {path}: {e2}")
                return []

    if ext == ".doc":
        try:
            return UnstructuredWordDocumentLoader(path).load()
        except Exception as e:
            print(f"[WARN] UnstructuredWordDocumentLoader failed for {path}: {e}")
            return []

    print(f"[WARN] Skipping unsupported file: {path}")
    return []

def load_all_documents(folder: str = DATA_DIR) -> List[Document]:
    docs: List[Document] = []
    for path in list_file_paths(folder):
        try:
            docs.extend(load_path_as_documents(path))
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
    return docs

def combine_documents_text(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

def init_faiss(docs: List[Document], force_rebuild: bool = False):
    global faiss_store

    def _split_docs(d: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        return splitter.split_documents(d) if d else []

    def _build_from_docs(d: List[Document]) -> FAISS:
        chunks = _split_docs(d)
        if not chunks:
            raise ValueError("No text chunks were produced from documents.")
        texts = [c.page_content for c in chunks]
        metas = [getattr(c, "metadata", {}) for c in chunks]
        vs = FAISS.from_texts(texts=texts, embedding=hf_embeddings, metadatas=metas)
        os.makedirs(FAISS_DIR, exist_ok=True)
        vs.save_local(folder_path=FAISS_DIR)
        return vs

    if force_rebuild:
        if not docs:
            raise ValueError("force_rebuild=True but no docs provided.")
        faiss_store = _build_from_docs(docs)
        return

    if os.path.exists(FAISS_DIR):
        try:
            faiss_store = FAISS.load_local(
                folder_path=FAISS_DIR,
                embeddings=hf_embeddings,
                allow_dangerous_deserialization=True,
            )
            return
        except Exception as e:
            if docs:
                print(f"[FAISS] Load failed ({e}). Rebuilding index…")
                faiss_store = _build_from_docs(docs)
                return
            raise RuntimeError("FAISS load failed and no docs to rebuild.") from e
    else:
        if not docs:
            raise ValueError("No FAISS index and no docs to create one.")
        faiss_store = _build_from_docs(docs)

def looks_uncertain(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in [
        "i don't know", "i do not know", "cannot answer", "not sure",
        "no information", "insufficient information", "lack of information"
    ])

def professional_clarification(query: str) -> str:
    return (
        "I’m not confident I can answer that from the current results. "
        f"Could you clarify what you’d like to know about “{query}”, or rephrase the question? "
        "For example, specify the aspect, time frame, or context."
    )

# --- Web Query Rewrite ---
class WebQueryRewrite(BaseModel):
    search_query: str = Field(..., description="Optimized web search query")
    rationale: str = Field(..., description="Brief reasoning for debugging")

_llm_rewrite = llm.with_structured_output(WebQueryRewrite)

def rewrite_query_for_web(user_query: str) -> WebQueryRewrite:
    system = (
        "Rewrite the user's text into a single, concise web search query suitable for DuckDuckGo/Google. "
        "Prefer key entities, optional helpful operators (quotes, site:, intitle:, filetype:) when clearly useful; "
        "avoid hallucinated constraints; keep it short."
    )
    user = f'Original query: "{user_query}"\nRewrite it into one optimal search query.'
    try:
        return _llm_rewrite.invoke([HumanMessage(content=system), HumanMessage(content=user)])
    except Exception:
        return WebQueryRewrite(search_query=user_query, rationale="LLM rewrite failed; used original query.")

def ddg_search(query: str) -> List[Dict[str, Any]]:
    tool = DuckDuckGoSearchResults(output_format="list")
    try:
        results = tool.invoke(query)
        return results if isinstance(results, list) else []
    except Exception:
        return []

def load_faiss_only() -> None:
    """Load an existing FAISS index into memory without rebuilding."""
    global faiss_store
    if not os.path.exists(FAISS_DIR):
        raise FileNotFoundError("FAISS index not found on disk.")
    faiss_store = FAISS.load_local(
        folder_path=FAISS_DIR,
        embeddings=hf_embeddings,
        allow_dangerous_deserialization=True,
    )

def rebuild_faiss_from_all_documents() -> int:
    """Rebuild the FAISS index from *all* supported docs currently in DATA_DIR."""
    docs = load_all_documents()
    if not docs:
        raise ValueError("No supported documents found in ./data to build embeddings.")
    init_faiss(docs, force_rebuild=True)
    return len(docs)

def delete_data_and_index() -> Dict[str, Any]:
    """Delete all supported files in DATA_DIR and remove the FAISS_DIR entirely."""
    global faiss_store
    os.makedirs(DATA_DIR, exist_ok=True)

    removed_files = []
    for p in list_file_paths(DATA_DIR):
        try:
            os.remove(p)
            removed_files.append(os.path.basename(p))
        except Exception as e:
            print(f"[WARN] Failed to remove {p}: {e}")

    index_removed = False
    if os.path.isdir(FAISS_DIR):
        try:
            shutil.rmtree(FAISS_DIR)
            index_removed = True
        except Exception as e:
            print(f"[WARN] Failed to remove FAISS dir: {e}")

    faiss_store = None
    return {"removed_files": removed_files, "index_removed": index_removed}

AGENT_NAME_MAP = {"summarize": "Agent1", "query": "Agent2", "internet": "Agent3"}

def _format_required(final: Dict[str, Any]) -> Dict[str, Any]:
    final = final or {}
    manager = final.get("manager_agent", {}) or {}
    agent_resps = final.get("agent_responses", {}) or {}

    selected_raw = manager.get("selected_agents") or manager.get("selected") or []
    selected_agents = [AGENT_NAME_MAP.get(x, x) for x in selected_raw]

    out = {
        "manager_agent": {
            "decision": manager.get("decision") or manager.get("justification") or "",
            "selected_agents": selected_agents,
        },
        "agent_responses": {}
    }

    if "Agent1" in agent_resps:
        a1 = agent_resps["Agent1"] or {}
        d = a1.get("data", {}) or {}
        out["agent_responses"]["Agent1"] = {
            "status": a1.get("status"),
            "data": {
                "document_summary": d.get("document_summary") or d.get("summary") or "",
                "keywords": d.get("keywords") or [],
            },
        }

    if "Agent2" in agent_resps:
        a2 = agent_resps["Agent2"] or {}
        d = a2.get("data", {}) or {}
        out["agent_responses"]["Agent2"] = {
            "status": a2.get("status"),
            "data": {
                "query": d.get("query") or d.get("user_query") or "",
                "response": d.get("response") or "",
            },
        }

    if "Agent3" in agent_resps:
        a3 = agent_resps["Agent3"] or {}
        d = a3.get("data", {}) or {}
        out["agent_responses"]["Agent3"] = {
            "status": a3.get("status"),
            "data": {
                "query": d.get("query") or d.get("user_query") or "",
                "response": d.get("response") or "",
                "source": d.get("source"),
            },
        }

    return out
