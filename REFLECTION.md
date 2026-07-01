# Reflection / Hallucination Prevention

## What is Reflection?

Reflection is a second-pass LLM verification step that runs immediately after the
initial answer is generated. It checks whether every claim in the answer is actually
supported by the retrieved policy excerpts — and removes or qualifies anything that isn't.

This pattern is used in production RAG systems (including the
[NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag)) to prevent the
most damaging class of RAG errors: **confident, fluent answers that cite policies or
facts that don't exist**.

---

## The Problem It Solves

Without reflection, the LLM occasionally:

- **Invents policy reference codes** that look real but aren't in SETU's documents
  (e.g. `SETU-001`, `SETU-HR-002`, `SETU-AC-005`)
- **Cites the wrong source policy** for a fact (e.g. attributes a harassment procedure
  to the Data Protection Policy instead of the Dignity and Respect Policy)
- **Fabricates specific numbers** (e.g. "staff are entitled to 15 days" when the policy
  says entitlements follow individual contracts)

These errors are particularly harmful in a compliance context — a staff member acting
on a hallucinated entitlement could face disciplinary or legal consequences.

---

## How It Works

```
User query
    │
    ▼
[Hybrid Retrieval]          ← BM25 + MMR + RRF + cross-encoder rerank
    │
    ▼
[Initial Answer Generation] ← LLM call #1: answer from context
    │
    ▼
[Reflection / Verification] ← LLM call #2: fact-check against context
    │
    ├── VERIFIED → return original answer unchanged
    │
    └── REVISED  → return corrected answer (unsupported claims removed)
```

### Reflection prompt (simplified)

```
Retrieved policy excerpts (ground truth):
[Source: Dignity and Respect Policy]
...excerpt text...

Draft answer to verify:
According to SETU-HR-002, staff should contact HR within 5 days...

Check the draft answer against the excerpts ONLY. Identify:
1. Policy names or reference codes NOT present in the excerpts
2. Specific numbers or dates NOT traceable to the excerpts
3. Any claim that goes beyond what the excerpts state

If fully grounded: VERIFIED: <original answer>
If unsupported claims found: REVISED: <corrected answer>
```

### Example

**Question:** *How does SETU handle workplace harassment complaints?*

**Initial answer (hallucinated):**
> According to SETU policy SETU-HR-005, all harassment complaints must be submitted
> within 10 working days to the HR department. The Dignity and Respect Policy
> (SETU-DR-001) outlines a three-stage process...

**After reflection (REVISED):**
> According to the **Dignity and Respect Policy**, SETU has a formal complaints
> procedure for workplace harassment. Staff should contact the relevant HR office.
> Refer to the full policy document for specific timescales and procedural steps.

The reflection removed:
- `SETU-HR-005` (not in any retrieved excerpt)
- `SETU-DR-001` (not in any retrieved excerpt)
- "10 working days" (not stated in the retrieved text)
- "three-stage process" (not stated in the retrieved text)

---

## Implementation

The reflection step lives in [information_agent.py](information_agent.py):

```python
def _reflect_and_verify(query, context, initial_answer, llm) -> str:
    reflect_prompt = (
        "You are a fact-checker for a university policy assistant.\n\n"
        "Retrieved policy excerpts (ground truth):\n"
        f"{context}\n\n"
        "Draft answer to verify:\n"
        f"{initial_answer}\n\n"
        "Check the draft answer against the excerpts ONLY. Identify:\n"
        "1. Policy names or reference codes NOT present in the excerpts\n"
        "2. Specific numbers, dates, or entitlements NOT traceable to the excerpts\n"
        "3. Any claim that contradicts or goes beyond what the excerpts state\n\n"
        "If fully grounded: VERIFIED: <original answer unchanged>\n"
        "If unsupported claims found: REVISED: <corrected answer>\n"
        "Output only VERIFIED or REVISED, nothing else."
    )
    reflection = llm.invoke(reflect_prompt).content.strip()
    if reflection.startswith("VERIFIED:"):
        return reflection[len("VERIFIED:"):].strip()
    elif reflection.startswith("REVISED:"):
        return reflection[len("REVISED:"):].strip()
    return initial_answer  # fallback: return original if unexpected output
```

`get_information()` calls it as the final step:

```python
initial_answer = llm.invoke(answer_prompt).content
return _reflect_and_verify(query, context, initial_answer, llm)
```

---

## Cost and Latency

| Step | LLM calls | Latency (Groq llama-3.1-8b) |
|------|-----------|----------------------------|
| Without reflection | 2 (query expand + answer) | ~1.5s |
| With reflection | 3 (query expand + answer + verify) | ~2.5s |

The extra ~1 second is worth it for compliance use cases where a hallucinated
policy reference could mislead staff or students.

---

## Inspired By

- [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag) — optional
  reflection step in the response generation pipeline
- [Self-RAG (Asai et al., 2023)](https://arxiv.org/abs/2310.11511) — retrieval and
  reflection tokens for grounded generation
- [RAGAS faithfulness metric](https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html)
  — measures what fraction of answer claims are supported by context
