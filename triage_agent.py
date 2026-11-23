"""
Triage agent — classifies an incoming query about SETU policies.

Determines:
  - which category the query belongs to
  - whether the information agent should retrieve policy context
  - which specific policies are most likely to be relevant
"""
from typing import TypedDict, List
from config.llm_config import get_llm


class TriageResult(TypedDict):
    urgency: str
    problem_types: List[str]
    needs_info: bool
    needs_action: bool
    info_query: str
    actions_needed: List[str]
    raw_message: str


ROUTING_TABLE = {
    "policy_lookup":      ["information"],
    "procedure_query":    ["information", "action"],
    "eligibility_check":  ["information", "action"],
    "comparison_query":   ["information"],
    "document_request":   ["action"],
    "out_of_scope":       [],
}

URGENCY_GUIDE = {
    "policy_lookup":     "low",
    "procedure_query":   "medium",
    "eligibility_check": "medium",
    "comparison_query":  "low",
    "document_request":  "low",
    "out_of_scope":      "low",
}

# Hint map: category -> likely source documents to prioritise
POLICY_HINTS = {
    "policy_lookup":     "any matching SETU policy document",
    "procedure_query":   "procedural sections within SETU policy documents",
    "eligibility_check": "leave, recruitment, student welfare, or equality policies",
    "comparison_query":  "multiple SETU policy documents",
    "document_request":  "document index",
    "out_of_scope":      "none",
}


# Keyword signals for zero-LLM triage routing
_KEYWORD_MAP: list[tuple[str, list[str]]] = [
    ("out_of_scope",      ["weather", "sport", "football", "news", "recipe", "joke",
                           "canteen", "bus", "timetable", "opening hours", "parking"]),
    ("document_request",  ["which document", "which policy", "where can i find",
                           "what document", "point me to", "link to"]),
    ("comparison_query",  ["difference between", "compare", "vs ", "versus",
                           "both policies", "which is stricter"]),
    ("eligibility_check", ["am i eligible", "do i qualify", "can i apply",
                           "am i entitled", "qualify for", "eligible for"]),
    ("procedure_query",   ["how do i", "how to", "what steps", "what is the process",
                           "how should i", "what do i need to do", "procedure for",
                           "report a", "submit a", "apply for"]),
]


def classify_message(message: str) -> TriageResult:
    """Classify a query using keyword rules first; fall back to LLM for ambiguous cases.

    Keyword routing avoids a full LLM call (~300ms + rate-limit quota) for the
    majority of queries where intent is unambiguous from the words alone.
    LLM fallback handles edge cases where keywords don't match clearly.
    """
    msg_lower = message.lower()

    # Keyword pass — O(n) scan, no API call
    category = None
    for cat, keywords in _KEYWORD_MAP:
        if any(kw in msg_lower for kw in keywords):
            category = cat
            break

    # LLM fallback for ambiguous cases (no keyword matched → likely policy_lookup)
    if category is None:
        # Most queries are policy lookups — only use LLM when it might be a
        # procedure or eligibility question without strong keyword signal.
        ambiguous_signals = ["can", "may", "must", "should", "allowed", "permitted",
                             "required", "need to", "have to", "when", "what happens"]
        if any(s in msg_lower for s in ambiguous_signals):
            try:
                llm = get_llm()
                prompt = (
                    "Classify this SETU policy query into ONE category:\n"
                    "- policy_lookup | procedure_query | eligibility_check | "
                    "comparison_query | document_request | out_of_scope\n\n"
                    f"Query: {message}\n\nReply with only the category name."
                )
                result = llm.invoke(prompt).content.strip().lower().replace("-", "_")
                category = result if result in ROUTING_TABLE else "policy_lookup"
            except Exception:
                category = "policy_lookup"
        else:
            category = "policy_lookup"

    routing        = ROUTING_TABLE[category]
    urgency        = URGENCY_GUIDE[category]
    info_query     = f"{message} — context: {POLICY_HINTS[category]}"
    actions_needed = [category] if "action" in routing else []

    return TriageResult(
        urgency=urgency,
        problem_types=[category],
        needs_info="information" in routing,
        needs_action="action" in routing,
        info_query=info_query,
        actions_needed=actions_needed,
        raw_message=message,
    )


if __name__ == "__main__":
    tests = [
        "What is SETU's data protection policy?",
        "How do I report a research misconduct allegation?",
        "Am I eligible for parental leave as a part-time employee?",
        "What is the difference between the EDI policy and the dignity and respect policy?",
        "Which document covers CCTV usage on campus?",
        "What time does the canteen open?",
    ]
    for msg in tests:
        r = classify_message(msg)
        print(f"Q: {msg}")
        print(f"   Category={r['problem_types'][0]}  Urgency={r['urgency']}  "
              f"NeedsInfo={r['needs_info']}  NeedsAction={r['needs_action']}")
        print()
