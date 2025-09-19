# Multi-Agents Chat

This project exposes a simple multi-agent Retrieval-Augmented Generation (RAG) service with optional web search. It’s built as a FastAPI app that orchestrates three task-focused agents via a lightweight LangGraph state machine, and it manages local document embeddings with FAISS. Below is a high-level explanation of how it all works so you can understand the approach and the request/response flow.

Also, the execution commands can be found at the end of this documentation. **Please add the GEMINI API KEY in the .env file.**

---

## 1 What this service does

* Hosts a REST API to answer questions about **your local files** and, when needed, the **web**.
* Decides—per query—**which agents to run**:

  1. *Summarize* a single local PDF,
  2. *RAG query* across your embedded documents,
  3. *Internet search* (DuckDuckGo snippets).
* Manages **short-term, per-thread memory** to rewrite follow-ups into a self-contained query.
* Builds/loads a **FAISS** index over supported files in `./data` and uses **bge-small** embeddings for retrieval.

---

## 2 Key components

### FastAPI app (`app.py`)

* Endpoints:

  * `GET /health` — health check
  * `POST /query` — main entrypoint; calls the graph with a `thread_id` and `query`, then returns a normalized payload.
  * `POST /embeddings/upload` — accepts supported files, saves them to `./data`, and **rebuilds** the FAISS index.
  * `DELETE /embeddings/reset` — deletes all files in `./data` and removes the FAISS index.
* Uses CORS middleware and a `GraphManager` instance to orchestrate requests.

### Graph manager (`helpers/graph_manager.py`)

* A two-node LangGraph:

  1. **`manager_node`**: resolves the user’s current message (optionally rewriting it using prior turn context), chooses agents, and records the decision.
  2. **`executor_node`**: runs the chosen agents in order and assembles the final response.
* Provides `GraphManager.invoke(thread_id, query)` which compiles and runs the graph with in-memory checkpointing for thread-scoped state.

### Agents (`helpers/agents.py`)

* **Agent1 — Summarize**: if there is **exactly one** PDF in `./data`, it loads it, concatenates text, and asks the LLM for a 3–4 sentence summary + keywords.
* **Agent2 — RAG Query**: loads an existing FAISS index, retrieves top chunks (k≈4) with bge embeddings, and asks the LLM to answer **using only retrieved snippets** (otherwise responds with a professional clarification).
* **Agent3 — Internet**: rewrites the query for the web, runs DuckDuckGo, and asks the LLM to answer **only from the returned snippets**; if uncertain, returns a clarification prompt.

### Utilities (`helpers/utils.py`)

* **LLM + Embeddings**: `ChatGoogleGenerativeAI` (Gemini 1.5 Flash) for reasoning and `BAAI/bge-small-en-v1.5` for embeddings.
* **Document I/O**: loaders for PDF/TXT/MD/CSV/TSV/XLSX/XLS/DOCX/DOC with a few resilient fallbacks.
* **FAISS lifecycle**: build (`init_faiss`, `rebuild_faiss_from_all_documents`), load (`load_faiss_only`), and delete (`delete_data_and_index`).
* **Web helpers**: query rewrite and DuckDuckGo search wrapper.
* **Response formatting**: normalizes agent outputs for the API response.

---

## 3 Request flow (end-to-end)

1. **Client calls** `POST /query` with `{ thread_id, query }`. The API forwards this to `GraphManager.invoke`.
2. **Manager node**:

   * **Rewrite with memory**: if the user’s message looks like a follow-up (“it/this/that…”) and there’s a previous query in this thread, the text is rewritten into a self-contained query.
   * **Route decision**: an LLM decides which agents to run (`summarize`, `query`, `internet`) based on the rewritten query and **how many PDFs** are present. Guardrails then enforce sensible choices (e.g., no “summarize” when multiple PDFs exist).
3. **Executor node**:

   * Runs the selected agents **in order**, collects their outputs, and builds a final payload that includes the manager’s decision and each agent’s result.
4. **API formats response** with `_format_required` so clients get a consistent envelope (`manager_agent`, `agent_responses`).

---

## 4 How each agent approaches the problem

### Agent1 — Single-PDF summarizer

* Requires **exactly one** PDF in `./data`; otherwise returns a 400-style message in its payload.
* Loads pages, concatenates text, prompts the LLM for a short summary and top keywords, and returns them.

### Agent2 — RAG over your documents

* **Prerequisite**: a FAISS index must already exist (built via `/embeddings/upload`).
* Loads the FAISS index; runs `similarity_search_with_score` (fallback to `similarity_search`) to retrieve top chunks; asks the LLM to answer **only** from those snippets.
* If the LLM output looks uncertain, it is converted into a **professional clarification** and marked **not answerable**.

### Agent3 — Internet (DuckDuckGo)

* Rewrites the user query for web search, fetches top results, and builds a snippets “evidence” block.
* Prompts the LLM to answer strictly from those snippets; otherwise returns a clarification.
* Returns the final answer plus a list of source URLs used.

---

## 5 Embeddings & indexing workflow

* **Supported types**: `.pdf, .txt, .md, .csv, .tsv, .xlsx, .xls, .docx, .doc`. Files go in `./data`.
* On `POST /embeddings/upload`, files are validated/saved, then **all** supported documents in `./data` are loaded, split (≈800 chars, 120 overlap), embedded with **bge-small-en-v1.5**, and indexed in **FAISS** at `./faiss_bge_index`.
* `/embeddings/reset` clears both the data folder and the FAISS index.

---

## 6 Short-term memory & safety rails

* **Per-thread memory**: the graph keeps a small history so the manager can rewrite ambiguous follow-ups into explicit queries.
* **Routing guardrails**: selections are deduped and adjusted to avoid impossible paths (e.g., “internet” only when no local PDFs exist).
* **Uncertainty detection**: helper functions down-rank vague LLM outputs and replace them with a concise clarification request.

---

## 7 Example lifecycle

1. Upload a few PDFs or docs to build the index: `POST /embeddings/upload` → FAISS index is created.
2. Ask a question: `POST /query` → Manager may choose **RAG** (Agent2).
3. Ask a timely or non-local question: Manager may add **Internet** (Agent3).
4. Ask for a summary when only one PDF is present: Manager includes **Summarize** (Agent1).

---

## 8 Notes & limits

* **One-PDF summary rule**: Agent1 runs only when exactly one PDF is detected in `./data`.
* **Index required for RAG**: Agent2 returns a helpful error if the FAISS index hasn’t been built yet.
* **LLM & embeddings choices**: Gemini 1.5 Flash is used for reasoning and structured outputs; BAAI/bge-small-en-v1.5 is used for embedding. Change these in `helpers/utils.py` if needed.

---

## 9 Minimal run instructions (local)

### Option A: Run directly with uvicorn
1. Put supported files in ./data (or use POST /embeddings/upload).
2. Start the API:
   ```bash
   uvicorn app:app --reload
   ```
3. Call the endpoints:
   * `GET /health`
   * `POST /embeddings/upload`
   * `POST /query` with `{ "thread_id": "demo", "query": "your question" }`
   * `DELETE /embeddings/reset` to start over.

### Option B: Run with Docker
1. Build the image:
   ```bash
   docker build -t rag-manager-api:latest .
   ```
2. Run the container:
#### For Windows Powershell
   ```bash
   docker run --rm -p 8000:8000 -v ${PWD}:/app --env-file .env rag-manager-api:latest
   ```

#### Linux / macOS (bash/zsh)
   ```bash
   docker run --rm -p 8000:8000 -v "${PWD}:/app" --env-file .env rag-manager-api:latest
   ```

#### Windows Command Prompt (cmd.exe)
   ```bash
   docker run --rm -p 8000:8000 -v "%cd%:/app" --env-file .env rag-manager-api:latest
   ```
---

## 10 Example cURL calls

### Health check
```bash
curl --location --request GET 'http://localhost:8000/health'
```

### Upload documents for embeddings
```bash
curl --location 'http://localhost:8000/embeddings/upload' \
--header 'Accept: application/json' \
--form 'files=@"./data/file1.pdf"' \
--form 'files=@"./data/file2.pdf"' \
--form 'files=@"./data/file3.pdf"'
```

### Query
```bash
curl --location 'http://localhost:8000/query' \
--header 'Content-Type: application/json' \
--data '{
    "thread_id": "demo-thread",
    "query": "What is Formula 1?"
  }'
```

### Reset embeddings
```bash
curl --location --request DELETE 'http://localhost:8000/embeddings/reset'
```

---