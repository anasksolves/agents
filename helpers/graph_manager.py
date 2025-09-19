from typing import TypedDict, List, Optional, Dict, Any, Tuple, Literal
import json
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from .utils import (
    llm,
    list_file_paths
)
from .agents import run_agent1_summarize, run_agent2_query, run_agent3_internet

class AgentState(TypedDict):
    messages: List[BaseMessage]
    query: Optional[str]
    resolved_query: Optional[str]
    task: str
    document: Optional[str]
    manager_decision: Optional[dict]
    selected_agents: List[str]
    final: Optional[dict]
    manager_history: List[dict]
    agent_outcomes: List[dict]

class RouteDecision(BaseModel):
    selected: List[Literal["summarize", "query", "internet"]]
    justification: str

_llm_route = llm.with_structured_output(RouteDecision)

def _llm_route_decision(query: str, pdf_count: int) -> dict:
    system = (
        "You are a Manager Agent that routes a user request to one or more agents, in order:\n"
        "- summarize: Summarize the content of exactly one local PDF.\n"
        "- query: Answer a question using one or more local PDFs (RAG).\n"
        "- internet: Answer using web search (DuckDuckGo).\n\n"
        "Return a structured decision with:\n"
        "selected: array of one or more of ['summarize','query','internet'] in execution order\n"
        "justification: a clear, professional explanation focusing on user intent and the tasks needed."
    )
    user = (
        f"USER_QUERY: {query}\n"
        f"PDF_COUNT_IN_DATA_DIR: {pdf_count}\n\n"
        "Guidance:\n"
        "- If the user wants a summary of a local PDF and pdf_count = 1, include 'summarize'.\n"
        "- If they want answers based on document(s) and pdf_count >=1 , include 'query'.\n"
        "- Include 'internet' only if the user wants latest info or question which should require browsing.\n"
    )
    try:
        decision: RouteDecision = _llm_route.invoke([HumanMessage(content=system), HumanMessage(content=user)])
        return {"selected": decision.selected, "justification": decision.justification}
    except Exception:
        if pdf_count == 0:
            return {"selected": ["internet"], "justification": "Fallback: external info likely needed."}
        if pdf_count == 1:
            return {"selected": ["summarize", "query"], "justification": "Fallback: summarize then answer from the PDF."}
        return {"selected": ["query"], "justification": "Fallback: answer from local PDFs."}

class MemoryRewrite(BaseModel):
    rewritten_query: str = Field(..., description="Single, concise query incorporating prior context if needed.")
    used_memory: bool = Field(..., description="True if previous context was used.")
    rationale: str = Field(..., description="Brief reason for rewrite (for logging/debug).")

_llm_memory_rewrite = llm.with_structured_output(MemoryRewrite)

def _rewrite_with_memory(prev_query: Optional[str], curr_query: str) -> MemoryRewrite:
    prev = (prev_query or "").strip()
    curr = (curr_query or "").strip()

    hint_words = [" it ", " its ", " that ", " this ", " they ", " them ", " those ", " these ", " such "]
    needs_memory = any(w in f" {curr.lower()} " for w in hint_words) or len(curr.split()) <= 4

    system = (
        "Rewrite the user's current message into a single, self-contained query. "
        "If it references a previous question (pronouns like it/this/that/they/etc.), incorporate prior context. "
        "Keep it short and free of hallucinated constraints."
    )
    user = (
        f"PREVIOUS_QUESTION:\n{prev or '(none)'}\n\n"
        f"CURRENT_MESSAGE:\n{curr}\n\n"
        "Return: rewritten_query (string), used_memory (boolean), rationale (string)."
    )
    try:
        if needs_memory and prev:
            return _llm_memory_rewrite.invoke([HumanMessage(content=system), HumanMessage(content=user)])
        else:
            return MemoryRewrite(rewritten_query=curr, used_memory=False, rationale="No prior context needed.")
    except Exception:
        return MemoryRewrite(rewritten_query=curr, used_memory=False, rationale="LLM rewrite failed; used current as-is.")

def _apply_guardrails(selection: List[str], pdf_count: int) -> Tuple[List[str], List[str]]:
    adjustments: List[str] = []
    ordered, seen = [], set()
    for s in selection:
        if s not in seen:
            ordered.append(s); seen.add(s)

    if pdf_count == 0:
        filtered = [s for s in ordered if s == "internet"]
        if not filtered:
            adjustments.append("Added 'internet' because no local PDFs are available.")
            filtered = ["internet"]
        return filtered, adjustments

    if pdf_count == 1:
        return [s for s in ordered if s in ("summarize", "query", "internet")], adjustments

    filtered = [s for s in ordered if s in ("query", "internet")]
    if not filtered:
        filtered, adj = ["query"], "Replaced 'summarize' with 'query' for multiple PDFs."
        adjustments.append(adj)
    return filtered, adjustments

def manager_node(state: AgentState):
    state.setdefault("messages", [])
    state.setdefault("manager_history", [])
    state.setdefault("agent_outcomes", [])
    state.setdefault("selected_agents", [])
    state.setdefault("task", "")
    state.setdefault("document", None)
    state.setdefault("manager_decision", None)
    state.setdefault("resolved_query", None)

    query = state.get("query") or ""
    prev_query = None
    if state["agent_outcomes"]:
        prev_query = state["agent_outcomes"][-1].get("resolved_query") or state["agent_outcomes"][-1].get("query")

    mem_rewrite = _rewrite_with_memory(prev_query, query)
    resolved_query = mem_rewrite.rewritten_query.strip() or query
    state["resolved_query"] = resolved_query

    pdf_count = len(list_file_paths())

    # Base choice
    llm_decision = _llm_route_decision(resolved_query, pdf_count)
    selection_raw = llm_decision.get("selected", [])
    justification = llm_decision.get("justification", "")

    # Guardrails
    selected, adjustments = _apply_guardrails(selection_raw, pdf_count)

    decision_blob = {
        "selected": selection_raw,
        "final_selected": selected,
        "justification": justification,
        "adjustments": adjustments,
        "query": query,
        "resolved_query": resolved_query,
        "memory_rewrite": {
            "used_memory": mem_rewrite.used_memory,
            "rationale": mem_rewrite.rationale,
        },
    }

    state["selected_agents"] = selected
    state["task"] = ",".join(selected) if selected else ""
    state["manager_decision"] = decision_blob

    hist = state.get("manager_history", [])
    hist.append(decision_blob)
    state["manager_history"] = hist

    state["messages"].append(HumanMessage(content=json.dumps({"manager_decision": decision_blob})))
    return state

def executor_node(state: AgentState):
    state.setdefault("messages", [])
    state.setdefault("manager_history", [])
    state.setdefault("agent_outcomes", [])

    selected = state.get("selected_agents", [])
    query = state.get("resolved_query") or state.get("query") or ""
    agent_responses: Dict[str, Any] = {}

    for agent in selected:
        if agent == "summarize":
            res = run_agent1_summarize(query)
            agent_responses["Agent1"] = res
        elif agent == "query":
            res = run_agent2_query(query)
            agent_responses["Agent2"] = res
        elif agent == "internet":
            res = run_agent3_internet(query)
            agent_responses["Agent3"] = res

    final_payload = {
        "user_query": state.get("query") or "",
        "resolved_query": query,
        "manager_agent": {
            "decision": state.get("manager_decision", {}).get("justification", ""),
            "selected_agents": state.get("manager_decision", {}).get("final_selected", []),
            "adjustments": state.get("manager_decision", {}).get("adjustments", []),
        },
        "agent_responses": agent_responses,
    }
    state["final"] = final_payload
    state["messages"].append(HumanMessage(content=json.dumps(final_payload, ensure_ascii=False)))

    outcomes = state.get("agent_outcomes", [])
    outcomes.append({
        "query": state.get("query") or "",
        "resolved_query": query,
        "agents_run": list(selected),
        "agent2_answerable": agent_responses.get("Agent2", {}).get("data", {}).get("answerable"),
        "agent3_used": "Agent3" in agent_responses and agent_responses["Agent3"]["status"] == "200",
    })
    state["agent_outcomes"] = outcomes
    return state

def _route_from_manager(_state: AgentState):
    return "executor_node"

# ---------- Graph Manager Class ----------
class GraphManager:
    def __init__(self):
        self.workflow = StateGraph(AgentState)
        self.workflow.add_node("manager_node", manager_node)
        self.workflow.add_node("executor_node", executor_node)

        self.workflow.set_entry_point("manager_node")
        self.workflow.add_conditional_edges("manager_node", _route_from_manager)
        self.workflow.add_edge("executor_node", END)

        self.checkpointer = InMemorySaver()
        self.compiled = self.workflow.compile(checkpointer=self.checkpointer)

    def invoke(self, thread_id: str, query: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Invoke the graph with a delta-state so prior memory (for this thread_id)
        is loaded and merged. Returns the final payload dict.
        """
        delta: Dict[str, Any] = {}
        if query is not None:
            delta["query"] = query

        config = {"configurable": {"thread_id": thread_id}}
        result_state = self.compiled.invoke(delta, config=config)
        return result_state.get("final")
