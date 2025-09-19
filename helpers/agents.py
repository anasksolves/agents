from typing import Dict, Any, List, Optional
import os
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Literal
from . import utils as U
from .utils import (
    llm,
    looks_uncertain, professional_clarification,
    rewrite_query_for_web, ddg_search,
    list_file_paths, load_path_as_documents, load_all_documents, combine_documents_text
)
class Agent1Summary(BaseModel):
    summary: str = Field(..., description="3–4 sentence summary")
    keywords: List[str] = Field(..., description="Top 5 keywords")

class Agent2Answer(BaseModel):
    response: str = Field(..., description="Final answer or clarification")
    is_answered: bool = Field(..., description="Whether the question is answered from snippets")

class Agent3Internet(BaseModel):
    response: str = Field(..., description="Concise answer based on web snippets")
    source: List[str] = Field(..., description="List of URLs used")

_llm_agent1 = llm.with_structured_output(Agent1Summary)
_llm_agent2 = llm.with_structured_output(Agent2Answer)
_llm_agent3 = llm.with_structured_output(Agent3Internet)

# ---------- Agent 1: Summarize PDF ----------
def run_agent1_summarize(query_text: Optional[str]) -> Dict[str, Any]:
    pdfs = list_file_paths()
    if len(pdfs) != 1:
        return {"status": "400", "data": {"user_query": query_text or "", "error": f"Agent1 expects exactly one PDF in ./data, found {len(pdfs)}."}}

    docs = load_path_as_documents(pdfs[0])
    doc_text = combine_documents_text(docs)
    prompt = (
        "Summarize in 3–4 sentences and extract the top 5 keywords.\n\n"
        f"{doc_text}\n\n"
        "Return fields: summary, keywords."
    )
    out: Agent1Summary = _llm_agent1.invoke([HumanMessage(content=prompt)])
    return {"status": "200", "data": {"user_query": query_text or "", "summary": out.summary, "keywords": out.keywords}}

# ---------- Agent 2: RAG over PDFs ----------
def run_agent2_query(query_text: Optional[str]) -> Dict[str, Any]:
    q = (query_text or "").strip()

    try:
        U.load_faiss_only()
    except FileNotFoundError:
        return {
            "status": "400",
            "data": {
                "user_query": q,
                "error": "Embeddings index not found. Upload files to /embeddings/upload to build embeddings first."
            },
        }
    except Exception as e:
        return {"status": "500", "data": {"user_query": q, "error": f"Failed to load FAISS index: {e}"}}

    if U.faiss_store is None:
        return {"status": "500", "data": {"user_query": q, "error": "FAISS store not initialized."}}

    # Retrieval
    retrieved, scores = [], []
    try:
        results = U.faiss_store.similarity_search_with_score(q, k=4)
        for d, s in results:
            retrieved.append(d.page_content)
            scores.append(s)
    except Exception:
        results = U.faiss_store.similarity_search(q, k=4)
        retrieved = [d.page_content for d in results]
        scores = []

    has_docs = len(retrieved) > 0
    strong_evidence = has_docs and (not scores or (min(scores) <= 1.0))

    prompt = (
        "You are given a user query and snippets retrieved from local PDFs. "
        "Answer the query using ONLY those snippets. "
        "If the snippets do not contain the necessary information, DO NOT invent an answer. "
        "Provide a brief, professional clarification asking the user to refine the question.\n\n"
        f"QUERY:\n{q}\n\n"
        f"RETRIEVED_SNIPPETS:\n{retrieved}\n\n"
        'Return fields: response (string), is_answered (boolean).'
    )
    out: Agent2Answer = _llm_agent2.invoke([HumanMessage(content=prompt)])
    resp_text = (out.response or "").strip()
    is_answered = bool(out.is_answered)

    if looks_uncertain(resp_text) or not resp_text:
        resp_text = professional_clarification(q)
        is_answered = False

    answerable = bool(is_answered and strong_evidence)
    return {"status": "200", "data": {"user_query": q, "response": resp_text, "is_answered": is_answered, "answerable": answerable}}

# ---------- Agent 3: Internet (DDG) ----------
def run_agent3_internet(query_text: Optional[str]) -> Dict[str, Any]:
    q_raw = (query_text or "").strip()
    if not q_raw:
        return {"status": "400", "data": {"user_query": "", "response": "Please provide a query to search.", "source": []}}

    rewrite = rewrite_query_for_web(q_raw)
    q_rewritten = (rewrite.search_query or q_raw).strip()

    results = ddg_search(q_rewritten)
    if not results:
        results = ddg_search(q_raw)

    top = results[:8]
    urls = [r.get("link") for r in top if r.get("link")]
    blocks = []
    for r in top:
        title = r.get("title", "") or ""
        link = r.get("link", "") or ""
        snippet = r.get("snippet", "") or ""
        blocks.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}")
    evidence = "\n---\n".join(blocks).strip()

    if not evidence:
        return {
            "status": "200",
            "data": {
                "user_query": q_raw,
                "response": professional_clarification(q_raw),
                "source": urls,
            },
        }

    prompt = (
        "Use ONLY the following DuckDuckGo snippets (titles, URLs, brief summaries) to answer the user’s question. "
        "If the needed information is not present, do NOT answer “I don't know.” "
        "Provide a brief, professional clarification asking the user to refine the question.\n\n"
        f"USER QUERY (original):\n{q_raw}\n\n"
        f"REWRITTEN SEARCH QUERY:\n{q_rewritten}\n\n"
        f"DDG SNIPPETS:\n{evidence}\n\n"
        "Return fields: response (string), source (list of URLs)."
    )
    out: Agent3Internet = _llm_agent3.invoke([HumanMessage(content=prompt)])
    response_text = out.response or ""
    if looks_uncertain(response_text):
        response_text = professional_clarification(q_raw)
    sources = out.source or urls

    return {
        "status": "200",
        "data": {
            "user_query": q_raw,
            "response": response_text,
            "source": list(dict.fromkeys(sources)),
            "rewrite": {"search_query": q_rewritten, "rationale": rewrite.rationale},
        },
    }
