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
COPY config/           ./config/
COPY app_gradio.py     .
COPY main_agent.py     .
COPY information_agent.py .
COPY triage_agent.py   .
COPY action_agent.py   .
COPY data_preprocessing.py .
COPY ingest.py         .

# ChromaDB vector store is mounted at runtime via volume
# Dataset JSONs for re-ingest are also mounted if needed
VOLUME ["/app/chroma_db", "/app/dataset"]

EXPOSE 7860

ENV LLM_PROVIDER=groq \
    LLM_MODEL=llama-3.1-8b \
    EMBEDDINGS_PROVIDER=jina \
    EMBEDDINGS_MODEL=jina-embeddings-v2-base-en

CMD ["python", "app_gradio.py"]
