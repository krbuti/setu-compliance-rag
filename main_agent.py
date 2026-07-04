"""
Main agent — LangGraph orchestrator for the SETU Compliance RAG chatbot.

Graph:
  triage -> information (if needs_info)
         -> action     (if needs_action)
         -> combine    -> END

Run interactively:
    python main_agent.py
"""
import concurrent.futures
from typing import TypedDict, Iterator
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from triage_agent import classify_message, TriageResult
from information_agent import load_vectorstore, get_information
from action_agent import run_action_agent
from config.llm_config import get_llm


class PolicyState(TypedDict):
    user_query: str
    triage_result: dict
    info_response: str
    action_response: str
    final_response: str


# ── Load vector store once at startup ────────────────────────────────────────
_vs = None

def _get_vs():
    global _vs
    if _vs is None:
        _vs = load_vectorstore()
    return _vs


# ── Graph nodes ───────────────────────────────────────────────────────────────

def triage_node(state: PolicyState) -> PolicyState:
    result = classify_message(state["user_query"])
    print(f"[Triage] category={result['problem_types'][0]}  "
          f"needs_info={result['needs_info']}  needs_action={result['needs_action']}")
    return {"triage_result": dict(result)}


def information_node(state: PolicyState) -> PolicyState:
    triage = state["triage_result"]
    if not triage["needs_info"]:
        return {"info_response": ""}
    print("[Information] Retrieving policy context...")
    answer = get_information(_get_vs(), state["user_query"])
    return {"info_response": answer}


def action_node(state: PolicyState) -> PolicyState:
    triage = state["triage_result"]
    if not triage["needs_action"]:
        return {"action_response": ""}
    category = triage["problem_types"][0]
    print(f"[Action] Running tools for category: {category}")
    result = run_action_agent(state["user_query"], category)
    return {"action_response": result}


_NO_INFO_REPLY = (
    "I'm sorry, I wasn't able to find relevant information in SETU's "
    "policy documents for your query. Please contact the relevant SETU "
    "office directly."
)


def _combine_prompt(query: str, info: str, action: str) -> str:
    sections = []
    if info:
        sections.append(f"Policy Information:\n{info}")
    if action:
        sections.append(f"Recommended Next Steps:\n{action}")
    return (
        "You are a SETU policy advisor answering a chat message.\n"
        "Rules:\n"
        "- Answer in plain prose, 2-4 sentences max\n"
        "- Do NOT write a letter (no 'Dear...', no 'Best regards', no sign-off)\n"
        "- Do NOT use placeholder text like [Your Name], [Staff Member], [Date]\n"
        "- Do NOT invent email addresses, phone numbers, or staff names — say 'contact SETU HR directly' instead\n"
        "- Cite the real policy document name if relevant\n"
        "- If the query names a policy code that does not appear in the information below, "
        "say clearly that SETU does not use numeric policy codes\n\n"
        f'Question: "{query}"\n\n'
        + "\n\n".join(sections)
    )


def combine_node(state: PolicyState) -> PolicyState:
    info  = state.get("info_response", "")
    action = state.get("action_response", "")
    query = state["user_query"]
    if not info and not action:
        return {"final_response": _NO_INFO_REPLY}
    final = get_llm().invoke(_combine_prompt(query, info, action)).content
    return {"final_response": final}


# ── Build the graph ───────────────────────────────────────────────────────────

workflow = StateGraph(PolicyState)
workflow.add_node("triage",      triage_node)
workflow.add_node("information", information_node)
workflow.add_node("action",      action_node)
workflow.add_node("combine",     combine_node)

workflow.set_entry_point("triage")
workflow.add_edge("triage",      "information")
workflow.add_edge("triage",      "action")
workflow.add_edge("information", "combine")
workflow.add_edge("action",      "combine")
workflow.add_edge("combine",     END)

app = workflow.compile()


def handle_query(message: str) -> str:
    result = app.invoke({"user_query": message})
    return result["final_response"]


def handle_query_stream(message: str) -> Iterator[str]:
    """Streaming variant: runs triage+retrieval then streams the combine answer.

    Triage is sequential (fast, ~1 LLM call). Information and action nodes
    run in parallel threads. The final combine LLM call streams tokens back
    to the caller so the Gradio UI updates incrementally.
    """
    state: PolicyState = {
        "user_query": message,
        "triage_result": {},
        "info_response": "",
        "action_response": "",
        "final_response": "",
    }
    state.update(triage_node(state))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        info_fut = executor.submit(information_node, dict(state))
        action_fut = executor.submit(action_node, dict(state))
        state.update(info_fut.result())
        state.update(action_fut.result())

    info = state.get("info_response", "")
    action = state.get("action_response", "")

    if not info and not action:
        yield _NO_INFO_REPLY
        return

    for chunk in get_llm().stream(_combine_prompt(message, info, action)):
        if chunk.content:
            yield chunk.content


# ── CLI loop ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SETU Compliance Policy Assistant")
    print("Type 'quit' to exit\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        print(f"\nAssistant: {handle_query(query)}\n")
