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


def classify_message(message: str) -> TriageResult:
    llm = get_llm()

    prompt = f"""You are a SETU (South East Technological University) policy assistant classifier.
Read the user query and classify it into exactly ONE of the following categories:

- policy_lookup       : User wants to know what a specific policy says
- procedure_query     : User wants to know how to do something (process / steps)
- eligibility_check   : User wants to know if they qualify for something
- comparison_query    : User wants to compare two or more policies
- document_request    : User wants to know which policy document to consult
- out_of_scope        : Query is unrelated to SETU institutional policies

Respond with ONLY the category name. No explanation, no punctuation.

User query: {message}
"""
    response = llm.invoke(prompt)
    category = response.content.strip().lower().replace("-", "_")

    if category not in ROUTING_TABLE:
        category = "policy_lookup"

    routing       = ROUTING_TABLE[category]
    urgency       = URGENCY_GUIDE[category]
    info_query    = f"{message} — context: {POLICY_HINTS[category]}"
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
