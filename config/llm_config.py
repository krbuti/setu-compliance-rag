import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
import requests

load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_anthropic import ChatAnthropic

LOCAL_EMBEDDING_MODELS = {
    "nomic": "nomic-embed-text",
}

REMOTE_EMBEDDING_MODELS = {
    "text-embedding-3-small": "text-embedding-3-small",
    "text-embedding-3-large": "text-embedding-3-large",
}

ANTHROPIC_MODELS = {
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-haiku":  "claude-haiku-4-5-20251001",
    "claude-opus":   "claude-opus-4-8",
}

# Groq free-tier models (fast LPU inference)
GROQ_MODELS = {
    "llama-3.1-8b":    "llama-3.1-8b-instant",
    "llama-3.3-70b":   "llama-3.3-70b-versatile",
    "llama-3.1-70b":   "llama-3.1-70b-versatile",
    "gemma2-9b":       "gemma2-9b-it",
    "mixtral-8x7b":    "mixtral-8x7b-32768",
}

# Cerebras models (wafer-scale fast inference, OpenAI-compatible)
CEREBRAS_MODELS = {
    "llama-3.1-8b":  "llama3.1-8b",
    "llama-3.3-70b": "llama3.3-70b",
}
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"


def _ollama_health(base_url: str) -> bool:
    try:
        requests.get(f"{base_url}/api/tags", timeout=3).raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


def get_embeddings(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> Embeddings:
    env_provider = os.getenv("EMBEDDINGS_PROVIDER", "local").lower()
    if provider is None:
        provider = env_provider

    openai_key = os.getenv("OPENAI_API_KEY")
    jina_key   = os.getenv("JINA_API_KEY")

    # Auto-promote: jina > openai > huggingface (if local Ollama not available)
    if provider == "local":
        if jina_key:
            print("[INFO] JINA_API_KEY found, switching embeddings to jina")
            provider = "jina"
        elif openai_key:
            print("[INFO] OpenAI key found, switching embeddings to openai")
            provider = "openai"
        else:
            print("[INFO] No Ollama/OpenAI available, switching embeddings to huggingface")
            provider = "huggingface"

    if model_name is None and provider == env_provider:
        model_name = os.getenv("EMBEDDINGS_MODEL")

    if provider == "local":
        model_name = model_name or "nomic"
        model_id = LOCAL_EMBEDDING_MODELS.get(model_name, model_name)
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        if not _ollama_health(base_url):
            if openai_key:
                print("[WARN] Ollama not available, falling back to OpenAI embeddings")
                return OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key, **kwargs)
            print("[WARN] Ollama not available, falling back to HuggingFace embeddings")
            return _make_hf_embeddings("nomic-ai/nomic-embed-text-v1.5", **kwargs)

        print(f"[OK] Ollama running at {base_url}")
        return OllamaEmbeddings(model=model_id, base_url=base_url, **kwargs)

    elif provider == "jina":
        if not jina_key:
            raise ValueError("JINA_API_KEY required for jina embeddings provider")
        # jina-embeddings-v2-base-en is 768-dim — compatible with ChromaDB
        model_id = model_name or os.getenv("EMBEDDINGS_MODEL") or "jina-embeddings-v2-base-en"
        return _make_jina_embeddings(model_id, jina_key)

    elif provider == "huggingface":
        # nomic-embed-text-v1.5 is 768-dim — matches existing ChromaDB
        model_id = model_name or os.getenv("EMBEDDINGS_MODEL") or "nomic-ai/nomic-embed-text-v1.5"
        return _make_hf_embeddings(model_id, **kwargs)

    elif provider == "openai":
        model_name = model_name or "text-embedding-3-small"
        model_id = REMOTE_EMBEDDING_MODELS.get(model_name, model_name)
        api_key = openai_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for openai embeddings provider")
        return OpenAIEmbeddings(model=model_id, api_key=api_key, **kwargs)

    else:
        raise ValueError(
            f"Unknown embeddings provider: {provider!r}. "
            "Must be 'local', 'jina', 'huggingface', or 'openai'"
        )


def _make_hf_embeddings(model_id: str, **kwargs):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError("Run: pip install langchain-huggingface einops")
    print(f"[OK] Using HuggingFace embeddings: {model_id}")
    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
        **kwargs,
    )


def _make_jina_embeddings(model_id: str, api_key: str):
    from langchain_core.embeddings import Embeddings
    import requests as _requests

    class JinaEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._embed(texts)

        def embed_query(self, text: str) -> list[float]:
            return self._embed([text])[0]

        def _clean(self, text: str) -> str:
            import re as _re
            # Remove control characters (except tab/newline) and truncate
            text = _re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
            return text[:8000]  # Jina max ~8192 tokens; 8000 chars is safe

        def _call_jina(self, batch: list[str]) -> list[list[float]]:
            import time as _time
            for attempt in range(4):
                try:
                    resp = _requests.post(
                        "https://api.jina.ai/v1/embeddings",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": model_id, "input": batch},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()["data"]
                    return [
                        item["embedding"]
                        for item in sorted(data, key=lambda x: x["index"])
                    ]
                except _requests.exceptions.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code == 400:
                        # 400 means bad content — fall through to per-item mode
                        raise
                    if attempt == 3:
                        raise
                    wait = 2 ** attempt
                    print(f"  [WARN] Jina attempt {attempt+1} failed ({exc}), retrying in {wait}s...")
                    _time.sleep(wait)
                except Exception as exc:
                    if attempt == 3:
                        raise
                    wait = 2 ** attempt
                    print(f"  [WARN] Jina attempt {attempt+1} failed ({exc}), retrying in {wait}s...")
                    _time.sleep(wait)

        def _embed(self, texts: list[str]) -> list[list[float]]:
            import time as _time
            cleaned = [self._clean(t) for t in texts]
            results = []
            batch_size = 32
            for i in range(0, len(cleaned), batch_size):
                batch = cleaned[i : i + batch_size]
                try:
                    results.extend(self._call_jina(batch))
                except _requests.exceptions.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code == 400:
                        # One bad item in batch — embed one-by-one to isolate it
                        print(f"  [WARN] Batch {i//batch_size} got 400, switching to per-item mode...")
                        for j, item_text in enumerate(batch):
                            try:
                                results.extend(self._call_jina([item_text]))
                            except Exception as item_exc:
                                print(f"  [WARN] Skipping chunk {i+j} (embed failed: {item_exc})")
                                results.append([0.0] * 768)  # zero vector placeholder
                    else:
                        raise
            return results

    print(f"[OK] Using Jina cloud embeddings: {model_id}")
    return JinaEmbeddings()


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs,
):
    env_provider = os.getenv("LLM_PROVIDER", "local").lower()
    if provider is None:
        provider = env_provider

    openai_key    = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    groq_key      = os.getenv("GROQ_API_KEY")
    cerebras_key  = os.getenv("CEREBRAS_API_KEY")

    # Auto-promote: groq > cerebras > anthropic > openai > local
    if provider == "local":
        if groq_key:
            print("[INFO] GROQ_API_KEY found, switching LLM to groq")
            provider = "groq"
        elif cerebras_key:
            print("[INFO] CEREBRAS_API_KEY found, switching LLM to cerebras")
            provider = "cerebras"
        elif anthropic_key:
            print("[INFO] ANTHROPIC_API_KEY found, switching LLM to anthropic")
            provider = "anthropic"
        elif openai_key:
            print("[INFO] OPENAI_API_KEY found, switching LLM to openai")
            provider = "openai"

    if model_name is None and provider == env_provider:
        model_name = os.getenv("LLM_MODEL")

    if provider == "local":
        model_name = model_name or "gemma3:1b"
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        if not _ollama_health(base_url):
            if groq_key:
                print("[WARN] Ollama not available, falling back to Groq")
                return _make_groq_llm(model_name, groq_key, **kwargs)
            if cerebras_key:
                print("[WARN] Ollama not available, falling back to Cerebras")
                return _make_cerebras_llm(model_name, cerebras_key, **kwargs)
            if anthropic_key:
                print("[WARN] Ollama not available, falling back to Anthropic")
                return ChatAnthropic(
                    model=ANTHROPIC_MODELS.get(model_name, "claude-sonnet-4-6"),
                    api_key=anthropic_key, **kwargs,
                )
            if openai_key:
                print("[WARN] Ollama not available, falling back to OpenAI")
                return ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_key, **kwargs)
            raise RuntimeError(f"Ollama not running at {base_url}. Run 'ollama serve'.")

        print(f"[OK] Ollama running at {base_url}")
        return ChatOllama(model=model_name, base_url=base_url, **kwargs)

    elif provider == "groq":
        if not groq_key:
            raise ValueError("GROQ_API_KEY required for groq provider")
        return _make_groq_llm(model_name, groq_key, **kwargs)

    elif provider == "cerebras":
        if not cerebras_key:
            raise ValueError("CEREBRAS_API_KEY required for cerebras provider")
        return _make_cerebras_llm(model_name, cerebras_key, **kwargs)

    elif provider == "openai":
        model_name = model_name or "gpt-3.5-turbo"
        if not openai_key:
            raise ValueError("OPENAI_API_KEY required for openai provider")
        return ChatOpenAI(model=model_name, api_key=openai_key, **kwargs)

    elif provider == "anthropic":
        model_name = model_name or "claude-sonnet-4-6"
        model_id = ANTHROPIC_MODELS.get(model_name, model_name)
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY required for anthropic provider")
        return ChatAnthropic(model=model_id, api_key=anthropic_key, **kwargs)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            "Must be 'local', 'groq', 'cerebras', 'openai', or 'anthropic'"
        )


def _make_groq_llm(model_name: Optional[str], api_key: str, **kwargs):
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError("Run: pip install langchain-groq")
    model_id = GROQ_MODELS.get(model_name or "llama-3.1-8b", model_name or "llama-3.1-8b-instant")
    print(f"[OK] Using Groq model: {model_id}")
    kwargs.setdefault("max_tokens", 200)
    return ChatGroq(model=model_id, api_key=api_key, **kwargs)


def _make_cerebras_llm(model_name: Optional[str], api_key: str, **kwargs):
    # Cerebras uses the OpenAI-compatible API
    model_id = CEREBRAS_MODELS.get(model_name or "llama-3.1-8b", model_name or "llama3.1-8b")
    print(f"[OK] Using Cerebras model: {model_id}")
    return ChatOpenAI(
        model=model_id,
        api_key=api_key,
        base_url=CEREBRAS_BASE_URL,
        **kwargs,
    )
