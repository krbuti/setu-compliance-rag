"""
Gradio chat interface for the SETU Compliance RAG chatbot.
Run locally:  python app_gradio.py
Live dashboard: /dashboard    Metrics API: /metrics
"""
import os
import time
import hashlib
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from main_agent import handle_query


# ── Metrics collector ─────────────────────────────────────────────────────────

class MetricsCollector:
    _ESCALATION_SIGNALS = [
        "no dedicated policy", "could not find", "outside the scope",
        "no specific policy", "don't have information", "not covered",
        "unable to find", "no policy found",
    ]

    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.total_queries = 0
        self.error_count = 0
        self.escalation_count = 0
        self._latencies: deque[float] = deque(maxlen=100)
        self._recent: deque[dict] = deque(maxlen=8)

    def record(self, question: str, answer: str, latency_ms: float, error: bool = False) -> None:
        with self._lock:
            self.total_queries += 1
            if error:
                self.error_count += 1
                self._recent.appendleft({
                    "time": time.strftime("%H:%M:%S"),
                    "snippet": question[:55] + ("…" if len(question) > 55 else ""),
                    "latency_ms": round(latency_ms),
                    "status": "error",
                })
                return
            self._latencies.append(latency_ms)
            escalated = any(s in answer.lower() for s in self._ESCALATION_SIGNALS)
            if escalated:
                self.escalation_count += 1
            self._recent.appendleft({
                "time": time.strftime("%H:%M:%S"),
                "snippet": question[:55] + ("…" if len(question) > 55 else ""),
                "latency_ms": round(latency_ms),
                "status": "escalated" if escalated else "ok",
            })

    def to_dict(self) -> dict:
        with self._lock:
            lats = list(self._latencies)
            uptime = round(time.time() - self.start_time)
            h, rem = divmod(uptime, 3600)
            m, s = divmod(rem, 60)
            return {
                "status": "live",
                "uptime_seconds": uptime,
                "uptime_label": f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s",
                "total_queries": self.total_queries,
                "avg_latency_ms": round(sum(lats) / len(lats)) if lats else 0,
                "min_latency_ms": round(min(lats)) if lats else 0,
                "max_latency_ms": round(max(lats)) if lats else 0,
                "error_count": self.error_count,
                "escalation_count": self.escalation_count,
                "error_rate_pct": round(self.error_count / self.total_queries * 100, 1) if self.total_queries else 0,
                "escalation_rate_pct": round(self.escalation_count / self.total_queries * 100, 1) if self.total_queries else 0,
                "recent_queries": list(self._recent),
                "system": {
                    "provider": os.getenv("LLM_PROVIDER", "groq"),
                    "model": os.getenv("LLM_MODEL", "llama-3.1-8b"),
                    "chunks": 2584,
                    "policies": 48,
                },
            }


collector = MetricsCollector()


# ── Answer cache ──────────────────────────────────────────────────────────────

class AnswerCache:
    """Exact-match TTL cache for repeated queries (demo-friendly)."""

    def __init__(self, ttl_minutes: int = 60, max_size: int = 200):
        self._data: dict[str, tuple[str, datetime]] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
        self._max_size = max_size
        self._lock = threading.Lock()

    def _key(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def get(self, query: str) -> str | None:
        k = self._key(query)
        with self._lock:
            if k in self._data:
                answer, ts = self._data[k]
                if datetime.now() - ts < self._ttl:
                    return answer
                del self._data[k]
        return None

    def set(self, query: str, answer: str) -> None:
        k = self._key(query)
        with self._lock:
            if len(self._data) >= self._max_size:
                oldest = min(self._data, key=lambda x: self._data[x][1])
                del self._data[oldest]
            self._data[k] = (answer, datetime.now())


_cache = AnswerCache()

_DISCLAIMER = (
    "\n\n---\n*Disclaimer: This is an academic AI prototype. "
    "Verify all information with official SETU policy documents or HR before acting on it.*"
)


def chat(message: str, history: list):
    if not message.strip():
        yield "Please enter a question about SETU policies."
        return

    # Cache hit → instant reply (no latency recorded as 0ms cache hit)
    cached = _cache.get(message)
    if cached:
        collector.record(message, cached, 0)
        yield cached + _DISCLAIMER
        return

    t0 = time.time()
    try:
        from main_agent import handle_query_stream
        partial = ""
        for chunk in handle_query_stream(message):
            partial += chunk
            yield partial
        latency = (time.time() - t0) * 1000
        collector.record(message, partial, latency)
        _cache.set(message, partial)
        yield partial + _DISCLAIMER
    except Exception:
        collector.record(message, "", (time.time() - t0) * 1000, error=True)
        raise


# ── FastAPI app + routes ──────────────────────────────────────────────────────

app = FastAPI(title="SETU Compliance RAG")

_DASHBOARD_PATH = Path(__file__).parent / "static" / "dashboard.html"


@app.get("/metrics", response_class=JSONResponse)
async def get_metrics():
    return collector.to_dict()


@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    return _DASHBOARD_PATH.read_text(encoding="utf-8")


# ── Gradio interface ──────────────────────────────────────────────────────────

demo = gr.ChatInterface(
    fn=chat,
    title="SETU Compliance Policy Assistant",
    description=(
        "Ask questions about SETU's 48 institutional policies — "
        "leave entitlements, recruitment, EDI, research conduct, data protection, and more.\n\n"
        "⚠️ **Academic prototype — not an official SETU service.**  \n"
        "Answers are generated using AI and may be incomplete or inaccurate. "
        "Always verify with official SETU policy documents or HR before acting on them."
    ),
    examples=[
        "How many weeks of maternity leave is a female staff member entitled to?",
        "What is SETU's data protection policy?",
        "Is Garda vetting required for all staff roles?",
        "What constitutes research misconduct at SETU?",
        "What support does SETU provide to staff with caring responsibilities?",
        "What does the Gen AI policy say about staff using AI tools?",
    ],
)

app = gr.mount_gradio_app(app, demo, path="/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from information_agent import load_vectorstore
    load_vectorstore()
    uvicorn.run(app, host="0.0.0.0", port=7860)
