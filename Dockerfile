FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (runtime only — no docling/pymupdf)
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Copy application source
COPY config/              ./config/
COPY app_gradio.py        .
COPY main_agent.py        .
COPY information_agent.py .
COPY triage_agent.py      .
COPY action_agent.py      .
COPY data_preprocessing.py .
COPY ingest.py            .

# Bake in the pre-built vector store and policy documents.
# No runtime ingest needed — chroma_db is read-only at query time.
COPY dataset/             ./dataset/
COPY chroma_db/           ./chroma_db/

# Set HF_HOME inside /app so the model cache lands in a writable location
# for OpenShift's arbitrary non-root UID (which cannot write to /.cache).
ENV HF_HOME=/app/.cache \
    HOME=/app \
    LLM_PROVIDER=groq \
    LLM_MODEL=llama-3.1-8b \
    EMBEDDINGS_PROVIDER=jina \
    EMBEDDINGS_MODEL=jina-embeddings-v2-base-en

# Pre-create HF cache directory so the cross-encoder downloads here at first startup.
# Build-time download is skipped — HuggingFace rate-limits unauthenticated build workers.
RUN mkdir -p /app/.cache

# OpenShift runs containers as a random non-root UID in group 0.
# uid=1001, gid=0 pattern: group-readable so any arbitrary UID works.
RUN useradd -u 1001 -r -g 0 -m -d /app appuser && \
    chown -R 1001:0 /app && \
    chmod -R g=u /app

USER 1001

CMD ["python", "app_gradio.py"]
