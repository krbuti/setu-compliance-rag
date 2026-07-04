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


# ── Answer cache (Redis-backed, in-memory fallback) ───────────────────────────

class AnswerCache:
    """Exact-match TTL cache backed by Upstash Redis when configured.

    Falls back to an in-memory dict when UPSTASH_REDIS_URL is not set.
    Redis mode survives pod restarts and is shared across HPA replicas.
    Set UPSTASH_REDIS_URL + UPSTASH_REDIS_TOKEN via oc set env to enable.
    """
    _PREFIX = "setu:answer:"

    def __init__(self, ttl_seconds: int = 3600, max_local: int = 200):
        self._ttl = ttl_seconds
        self._max_local = max_local
        self._lock = threading.Lock()
        self._local: dict[str, tuple[str, datetime]] = {}
        self._r = None

        redis_url = os.getenv("UPSTASH_REDIS_URL")
        if redis_url:
            try:
                import redis as _redis
                token = os.getenv("UPSTASH_REDIS_TOKEN", "")
                self._r = _redis.from_url(
                    redis_url,
                    password=token or None,
                    decode_responses=True,
                    socket_connect_timeout=3,
                )
                self._r.ping()
                print("[OK] Redis answer cache connected")
            except Exception as exc:
                print(f"[WARN] Redis unavailable ({exc}) — using in-memory cache")
                self._r = None

    def _key(self, query: str) -> str:
        return self._PREFIX + hashlib.md5(query.strip().lower().encode()).hexdigest()

    def get(self, query: str) -> str | None:
        k = self._key(query)
        if self._r:
            try:
                return self._r.get(k)
            except Exception:
                pass
        with self._lock:
            if k in self._local:
                answer, ts = self._local[k]
                if datetime.now() - ts < timedelta(seconds=self._ttl):
                    return answer
                del self._local[k]
        return None

    def set(self, query: str, answer: str) -> None:
        k = self._key(query)
        if self._r:
            try:
                self._r.setex(k, self._ttl, answer)
                return
            except Exception:
                pass
        with self._lock:
            if len(self._local) >= self._max_local:
                oldest = min(self._local, key=lambda x: self._local[x][1])
                del self._local[oldest]
            self._local[k] = (answer, datetime.now())


_cache = AnswerCache()


def _parse_history(raw_history) -> list[tuple[str, str]]:
    """Convert Gradio ChatInterface history to (user, assistant) pairs.

    Strips the disclaimer suffix from assistant messages so it doesn't
    pollute the context window sent to the LLM.
    """
    pairs = []
    for item in (raw_history or []):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg = str(item[0] or "").strip()
            asst_msg = str(item[1] or "").split("\n\n---\n")[0].strip()
            if user_msg and asst_msg:
                pairs.append((user_msg, asst_msg))
    return pairs[-3:]  # keep last 3 turns max


_DISCLAIMER = (
    "\n\n---\n*Disclaimer: This is an academic AI prototype. "
    "Verify all information with official SETU policy documents or HR before acting on it.*"
)


def chat(message: str, history: list):
    if not message.strip():
        yield "Please enter a question about SETU policies."
        return

    parsed_history = _parse_history(history)

    # Cache only for standalone queries (no conversation context)
    if not parsed_history:
        cached = _cache.get(message)
        if cached:
            collector.record(message, cached, 0)
            yield cached + _DISCLAIMER
            return

    t0 = time.time()
    try:
        from main_agent import handle_query_stream
        partial = ""
        for chunk in handle_query_stream(message, parsed_history):
            partial += chunk
            yield partial
        latency = (time.time() - t0) * 1000
        collector.record(message, partial, latency)
        if not parsed_history:
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
