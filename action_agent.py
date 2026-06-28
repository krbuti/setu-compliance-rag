"""
Action agent — SETU-specific tools for policy navigation.

Tools available:
  - list_related_policies : Given a topic, list SETU policies that cover it
  - get_policy_scope      : Return what population/area a policy applies to
  - find_responsible_party: Identify who is responsible for a policy area
  - suggest_next_steps    : Suggest practical next steps based on a query type
"""
from pathlib import Path
from langchain_core.tools import tool
from config.llm_config import get_llm

PDF_DIR = Path(__file__).parent.parent / "All Data"

# Index of all 48 policy documents with brief descriptions
POLICY_INDEX = {
    "Academic Unit Reviews Policy":             "Reviews of academic units and departments",
    "Acceptable Usage Policy":                  "Acceptable use of SETU IT systems and resources",
    "Authorship Policy":                        "Rules for academic authorship and publication credit",
    "Brand guidelines":                         "SETU visual identity and brand usage",
    "Capital Projects Management Policy":       "Managing capital investment and construction projects",
    "Closed Circuit Television (CCTV) Policy":  "Use of CCTV cameras across SETU campuses",
    "Co-branding and labelling guidelines":     "External partnership branding rules",
    "Code of Conduct for Responsible Research Practice": "Ethics and conduct for researchers",
    "Conflict of Interest Policy":              "Declaring and managing conflicts of interest",
    "Consultancy Policy":                       "Rules for staff undertaking external consultancy",
    "Data Governance Policy":                   "Governance of SETU's data assets",
    "Data Protection Policy":                   "GDPR compliance and personal data handling",
    "Data Retention Policy":                    "How long SETU retains different categories of data",
    "Dignity and Respect Policy for Students":  "Student rights to dignity and freedom from harassment",
    "Email Policy":                             "Rules for use of SETU email systems",
    "Energy Policy":                            "SETU's energy use and sustainability commitments",
    "Equality Statement":                       "SETU's statement on equality principles",
    "Equality, Diversity & Inclusion Policy":   "EDI policy covering all protected characteristics",
    "Fitness _ Preparedness to Practise Policy": "Staff fitness and preparedness to practise",
    "Fitness to Continue in Study Policy":      "Student fitness to continue their programme",
    "Fitness to Practise Policy & Procedure (Department of Pharmacy)": "Pharmacy-specific fitness standards",
    "Garda Vetting Policy":                     "Requirements for Garda (police) vetting",
    "Gender Identity & Expression Policy":      "Support and rights for gender identity expression",
    "Global Engagement Strategic Plan_ Global Minds, Global and Local Impact (2024-2028)": "SETU internationalisation strategy",
    "Honorary Degree Process":                  "Process for awarding honorary degrees",
    "Identity and naming conventions":          "Rules for SETU identity and naming",
    "Intellectual Property Policy":             "Ownership of IP created at SETU",
    "Leave Management Policy":                  "All categories of staff leave entitlements",
    "Master of Pharmacy Code of Conduct":       "MPharm student conduct standards",
    "Master of Pharmacy Open Disclosure Policy": "Open disclosure rules for MPharm programme",
    "Probation Policy":                         "Staff probation periods and reviews",
    "Procedures for Managing Allegations of Misconduct in Research": "Research misconduct investigation process",
    "Professional Service Unit Reviews Policy": "Reviews of professional service units",
    "Programme Development and Validation Policy": "Creating and validating academic programmes",
    "Progression from assistant lecturer to lecturer policy": "Promotion pathway for academic staff",
    "Quality Framework":                        "SETU's quality assurance framework",
    "Recruitment & Selection Policy":           "Hiring process and selection criteria",
    "Risk Management Policy":                   "Identifying and managing institutional risks",
    "SETU Climate Action Roadmap":              "SETU's sustainability and climate commitments",
    "SETU Staff Guidelines on the Use of Gen AI": "Staff guidance on using generative AI tools",
    "Supporting Employees with Caring Responsibilities Policy": "Support for staff with caring duties",
    "Tone and writing style guide":             "Communication and writing standards",
    "Treasury Management Policy":               "SETU's financial treasury management",
    "University Reviews Policy":                "Overall university review framework",
    "Use of Animals for Research and Teaching Policy": "Ethics for using animals in research",
    "Video production guidelines":              "Standards for SETU video content",
    "Visiting Academic Policy":                 "Rules for visiting academics at SETU",
    "Voluntary Campus Transfer Policy":         "Staff and student voluntary campus transfers",
}


@tool
def list_related_policies(topic: str) -> str:
    """Given a topic or keyword, return SETU policy documents most likely relevant to it."""
    topic_lower = topic.lower()
    matches = [
        f"- {name}: {desc}"
        for name, desc in POLICY_INDEX.items()
        if any(word in (name + " " + desc).lower() for word in topic_lower.split())
    ]
    if matches:
        return f"Policies related to '{topic}':\n" + "\n".join(matches[:8])
    return f"No direct policy match found for '{topic}'. Consider searching: Data Protection, Equality, Research, or Leave policies."


@tool
def get_policy_scope(policy_name: str) -> str:
    """Return a brief description of what a named SETU policy covers."""
    for name, desc in POLICY_INDEX.items():
        if policy_name.lower() in name.lower():
            return f"{name}: {desc}"
    return f"Policy '{policy_name}' not found in the index. Available policies: {', '.join(list(POLICY_INDEX.keys())[:10])}..."


@tool
def find_responsible_party(query: str) -> str:
    """Identify which SETU office or role is typically responsible for the topic in the query."""
    topic = query.lower()
    if any(w in topic for w in ["data", "gdpr", "personal", "retention"]):
        return "Data Protection Officer (DPO) — contact via dataprotection@setu.ie"
    if any(w in topic for w in ["research", "ethics", "misconduct", "authorship"]):
        return "Research Office / Research Ethics Committee"
    if any(w in topic for w in ["leave", "probation", "recruitment", "staff", "hr"]):
        return "Human Resources (HR) Department"
    if any(w in topic for w in ["student", "dignity", "fitness to continue", "appeal"]):
        return "Student Services / Academic Registry"
    if any(w in topic for w in ["equality", "diversity", "gender", "edi", "discrimination"]):
        return "Equality, Diversity & Inclusion Office"
    if any(w in topic for w in ["cctv", "security", "campus"]):
        return "Campus Security / Facilities Management"
    if any(w in topic for w in ["ip", "intellectual property", "consultancy"]):
        return "Research & Innovation Office"
    return "Refer to the relevant policy document or contact SETU's main office."


@tool
def suggest_next_steps(category: str) -> str:
    """Given a triage category, suggest practical next steps for the user."""
    steps = {
        "policy_lookup":     "1. Review the relevant policy PDF in the SETU policy library.\n2. Contact the responsible office if clarification is needed.",
        "procedure_query":   "1. Follow the procedure outlined in the relevant policy.\n2. Contact the responsible office to initiate the formal process.\n3. Keep records of all communications.",
        "eligibility_check": "1. Review the eligibility criteria in the relevant policy.\n2. Complete any required application forms.\n3. Submit to HR or the relevant office with supporting evidence.",
        "comparison_query":  "1. Read both policies side by side.\n2. Note where they overlap or conflict.\n3. Contact the Policy & Governance office for clarification on precedence.",
        "document_request":  "1. Locate the policy in SETU's online policy library.\n2. Check the version date to ensure you have the current version.",
        "out_of_scope":      "This query may not be covered by SETU's institutional policies. Try the SETU website or contact the relevant department directly.",
    }
    return steps.get(category, "Contact the relevant SETU office for guidance.")


def run_action_agent(query: str, category: str = "policy_lookup") -> str:
    """Run the action agent with available SETU tools."""
    from langgraph.prebuilt import create_react_agent

    llm   = get_llm()
    tools = [list_related_policies, get_policy_scope, find_responsible_party, suggest_next_steps]

    agent = create_react_agent(
        model=llm,
        tools=tools,
    )
    system = (
        "You are a SETU policy navigation assistant. "
        "Use the available tools to help the user find the right policies, "
        "identify responsible offices, and understand what to do next. "
        "Be concise and practical."
    )
    try:
        result = agent.invoke({
            "messages": [
                ("system", system),
                ("human", f"Query: {query}\nQuery category: {category}"),
            ]
        })
        return result["messages"][-1].content
    except Exception as exc:
        return f"Action agent unavailable: {exc}"


if __name__ == "__main__":
    print(list_related_policies.invoke("data protection"))
    print()
    print(find_responsible_party.invoke("I need to report research misconduct"))
    print()
    print(suggest_next_steps.invoke("procedure_query"))
