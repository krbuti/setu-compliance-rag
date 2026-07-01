# SETU Compliance Policy RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-purple)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5.8-orange)
![Docling](https://img.shields.io/badge/Docling-IBM-red)
![Groq](https://img.shields.io/badge/LLM-Groq-black)
![Jina](https://img.shields.io/badge/Embeddings-Jina_AI-blue)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)

> Multi-agent RAG chatbot over 48 SETU institutional policy PDFs.
> Runs fully on **free cloud APIs** (Groq + Jina AI) — no local GPU required.
> Includes a one-click Google Colab notebook with a shareable Gradio chat UI.

---

## Quick Start (Google Colab)

Open the notebook in Colab and run cells top-to-bottom:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krbuti/setu-compliance-rag/blob/Latest/colab_demo.ipynb)

You need two free API keys (add them as Colab Secrets):

| Secret name | Free tier | Sign up |
|-------------|-----------|---------|
| `GROQ_API_KEY` | 14,400 requests/day | https://console.groq.com/keys |
| `JINA_API_KEY` | 1M tokens free | https://jina.ai |

The notebook clones the repo, installs packages, builds ChromaDB (~2 min), then launches a Gradio chat UI with a public 72-hour share link.

---

## Architecture

```
User Query
    |
    v
[Triage Agent]
  Classifies: policy_lookup / procedure_query /
              eligibility_check / comparison_query / out_of_scope
    |
    +---> [Information Agent]           +---> [Action Agent]
    |       Hybrid retrieval:           |       LangGraph ReAct tools:
    |       - Self-query filter         |       - list_related_policies
    |       - BM25 (keyword)            |       - get_policy_scope
    |       - MMR ChromaDB (semantic)   |       - find_responsible_party
    |       - RRF fusion                |       - suggest_next_steps
    |       - Cross-encoder rerank      |
    |            |                      |
    +---> [Combine Agent] <------------+
              |
              v
         Final Response
         (grounded, cited, professional)
```

**PDF parsing + indexing pipeline:**
```
48 SETU PDFs
    |
    Docling DocumentConverter
    (AI layout analysis: headings, tables, lists preserved as Markdown)
    |
    Table-aware chunking (NEW)
    - Markdown table blocks kept ATOMIC — never split across chunks
    - Nearest section heading prepended to each table chunk for context
    - Non-table text split at child=256 chars / parent=1024 chars
    |
    Jina AI cloud embeddings (jina-embeddings-v2-base-en, 768-dim)
    |
    ChromaDB (persistent, on-disk, 2,584 chunks from 48 documents)
```

**Retrieval pipeline (per query):**
```
Query
  -> Self-query filter     (detect named policy -> add metadata filter)
  -> Lightweight expand    (1 LLM rewrite)
  -> BM25 retrieval        (k=8 keyword matches)
  -> MMR ChromaDB          (k=8 semantic, diversity-aware)
  -> RRF fusion            (merge both ranked lists)
  -> Cross-encoder rerank  (ms-marco-MiniLM, ~50ms, no API call)
  -> Top-5 chunks
  -> Initial answer generation  (LLM call #1, grounded in context)
  -> Reflection / verification  (LLM call #2, removes hallucinated claims)
  -> Final cited answer
```

---

## Project Structure

```
setu_compliance_rag/
  config/
    llm_config.py          # LLM + embeddings factory (Groq/Cerebras/Anthropic/OpenAI/local)
  dataset/                 # 48 JSON files — one per PDF (committed to git)
  chroma_db/               # Built by ingest.py (gitignored)
  build_dataset.py         # Step 1: Docling PDF -> dataset/*.json  (run once)
  ingest.py                # Step 2: dataset/*.json -> ChromaDB     (resumable)
  data_preprocessing.py    # Table-aware parent-child chunking
  triage_agent.py          # Query classifier (6 SETU-specific categories)
  information_agent.py     # Production hybrid retrieval pipeline
  action_agent.py          # LangGraph ReAct tools for policy navigation
  main_agent.py            # LangGraph StateGraph orchestrator
  app_gradio.py            # Gradio chat UI (local or Colab)
  colab_demo.ipynb         # Google Colab notebook (6-step setup + Gradio launch)
  requirements.txt
  .env.example
```

---

## Local Setup

> **Important:** Run all commands from the `setu_compliance_rag/` folder.

### 1. Create virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```powershell
copy .env.example .env
```

Edit `.env` — minimum required for cloud mode:

```dotenv
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b
GROQ_API_KEY=gsk_...

EMBEDDINGS_PROVIDER=jina
EMBEDDINGS_MODEL=jina-embeddings-v2-base-en
JINA_API_KEY=jina_...
```

Supported providers:

| Provider | `LLM_PROVIDER` value | Key env var |
|----------|---------------------|-------------|
| Groq (free) | `groq` | `GROQ_API_KEY` |
| Cerebras (free) | `cerebras` | `CEREBRAS_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Ollama (local) | `local` | — |

If `LLM_PROVIDER=local` but Ollama is not running, the system auto-promotes to the first available cloud provider (groq → cerebras → anthropic → openai).

### 3. Build ChromaDB

The `dataset/` JSON files are already committed — no PDF parsing needed:

```powershell
python ingest.py
```

Embeds 2,584 chunks via Jina AI (~2 minutes). Supports resume — if interrupted, re-running skips already-indexed chunks.

To force a full re-extraction from PDFs (requires `All Data/` folder with the original PDFs):

```powershell
python build_dataset.py          # skip existing JSONs
python build_dataset.py --force  # re-extract all 48 PDFs
```

### 4. Run

**CLI chatbot:**
```powershell
python main_agent.py
```

**Gradio web UI (local):**
```powershell
python app_gradio.py
# Open http://127.0.0.1:7860
```

---

## What's New (Latest branch)

### Reflection / Hallucination Prevention

A second-pass LLM verification step now runs after every answer is generated.
It checks whether every claim — especially policy names, reference codes, and
specific numbers — is actually present in the retrieved context.

```
[Initial Answer] → [Reflection Check] → VERIFIED (unchanged) or REVISED (hallucinations removed)
```

**Without reflection:** The LLM sometimes invents policy reference codes like
`SETU-001` or `SETU-HR-005` that look real but don't exist in any SETU document.

**With reflection:** Those invented codes are caught and removed before the answer
reaches the user.

See [REFLECTION.md](REFLECTION.md) for full details, examples, and implementation notes.

### Table-aware chunking
Previous versions split markdown tables at 256-char boundaries, tearing table rows away from their header row. Answers about leave days, fee structures, and approval thresholds were vague as a result.

The new `data_preprocessing.py` detects markdown table blocks and keeps them **atomic** — one table = one chunk, no matter the size. The nearest section heading is prepended so the model understands column semantics:

```
## Annual Leave Entitlements        <-- heading prepended for context

| Grade      | Days per Year |
|------------|--------------|
| Clerical   | 22           |
| Academic   | 30           |
```

This reduced total chunk count from 3,394 → 2,584 (810 fewer, because table rows are no longer split into individual chunks).

### Resilient Jina embeddings
- Batch size reduced 128 → 32 (avoids API timeouts)
- Timeout increased 60 → 120 seconds
- 4-attempt exponential backoff on transient failures
- Text sanitized (control chars removed, truncated at 8,000 chars)
- On `400 Bad Request`, falls back to per-item embedding with zero-vector placeholder for unembeddable chunks

### Resume-from-checkpoint ingest
`ingest.py` now checks which chunk IDs already exist in ChromaDB before indexing. If the process is interrupted mid-run, re-running it picks up exactly where it left off.

### Gradio chat UI + Google Colab notebook
- `app_gradio.py` — standalone Gradio interface, works locally or in Colab
- `colab_demo.ipynb` — 6-cell notebook: clone → install → API keys → ingest → load agent → launch Gradio with public share link

---

## Key Design Choices vs Lab-06 Baseline

| Aspect | Lab-06 | This project |
|--------|--------|--------------|
| Data source | Inline text strings | 48 PDF files via Docling |
| Chunking | Fixed-size splitter | Table-aware parent-child (256/1024) |
| Retrieval | Single similarity_search | BM25 + MMR ChromaDB + RRF |
| Reranking | LLM-based (slow, expensive) | Cross-encoder (local, ~50ms) |
| Query expansion | 3 LLM rewrites | 1 lightweight rewrite |
| Self-query | None | Named-policy metadata filter |
| Agent framework | None | LangGraph StateGraph |
| LLM provider | Local Ollama only | Groq / Cerebras / Anthropic / OpenAI / Ollama |
| Embedding provider | Local Ollama only | Jina AI / HuggingFace / OpenAI / Ollama |
| UI | Terminal only | Gradio (local + Colab shareable link) |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ChromaDB not found` | Run `python ingest.py` |
| `GROQ_API_KEY missing` | Add to `.env` or Colab Secrets |
| Ingest times out mid-run | Re-run `python ingest.py` — it resumes from last checkpoint |
| Docling first run is slow | Normal — downloads DocLayNet + TableFormer models (~500 MB), cached after |
| Gradio share link fails | Local network restriction; use `http://127.0.0.1:7860` instead |
| Ollama not running | Either start `ollama serve` or add a Groq/Cerebras key — auto-promotes |

---

*Built as part of the SETU Building AI Systems module — SETU Compliance project.*
