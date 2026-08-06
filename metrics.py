"""
Prometheus metrics module for RAG Research Assistant.

Exposes:
  - HTTP request latency & count (via middleware)
  - Query-specific latency histogram
  - Query accuracy (confidence + grounding)
  - Document upload counter
  - Active user gauge
"""

import time
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, PlainTextResponse


# ============================================================================
# Lightweight Prometheus client (zero external deps)
# ============================================================================

class Counter:
    """Monotonically increasing counter with optional labels."""

    def __init__(self, name: str, help_text: str, labels: list[str] | None = None):
        self.name = name
        self.help = help_text
        self.labels = labels or []
        self._values: dict[tuple, float] = {}

    def inc(self, amount: float = 1.0, **kwargs):
        key = tuple(kwargs.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0) + amount

    def collect(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        for key, val in self._values.items():
            label_str = self._label_str(key)
            lines.append(f"{self.name}{label_str} {val}")
        return "\n".join(lines)

    def _label_str(self, key: tuple) -> str:
        if not self.labels:
            return ""
        pairs = ",".join(f'{l}="{v}"' for l, v in zip(self.labels, key))
        return "{" + pairs + "}"


class Gauge:
    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help = help_text
        self._value = 0.0

    def set(self, value: float):
        self._value = value

    def inc(self, amount: float = 1.0):
        self._value += amount

    def dec(self, amount: float = 1.0):
        self._value -= amount

    def collect(self) -> str:
        return (
            f"# HELP {self.name} {self.help}\n"
            f"# TYPE {self.name} gauge\n"
            f"{self.name} {self._value}"
        )


class Histogram:
    """Fixed-bucket histogram compatible with Prometheus exposition format."""

    def __init__(self, name: str, help_text: str, buckets: list[float] | None = None):
        self.name = name
        self.help = help_text
        self.buckets = buckets or [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._bucket_counts = {b: 0 for b in self.buckets}
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float):
        self._sum += value
        self._count += 1
        for b in self.buckets:
            if value <= b:
                self._bucket_counts[b] += 1

    def collect(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        cumulative = 0
        for b in self.buckets:
            cumulative += self._bucket_counts[b]
            lines.append(f'{self.name}_bucket{{le="{b}"}} {cumulative}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self._count}')
        lines.append(f"{self.name}_sum {self._sum}")
        lines.append(f"{self.name}_count {self._count}")
        return "\n".join(lines)


# ============================================================================
# Metric instances
# ============================================================================

# HTTP-level (populated by middleware)
HTTP_REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labels=["method", "endpoint", "status"],
)
HTTP_REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
)

# Query-level
QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "End-to-end RAG query latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
QUERY_COUNT = Counter("rag_query_total", "Total RAG queries processed")
QUERY_CONFIDENCE = Histogram(
    "rag_query_confidence",
    "Confidence score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
GROUNDED_ANSWERS = Counter(
    "rag_grounded_answers_total",
    "Answers that passed grounding verification",
    labels=["grounded"],
)

# Resource-level
DOCUMENTS_LOADED = Gauge("rag_documents_loaded", "Number of document chunks in vector store")
DOCUMENT_UPLOADS = Counter("rag_document_uploads_total", "Total documents uploaded")
ACTIVE_USERS = Gauge("rag_active_users", "Total registered users")

# ============================================================================
# Collector registry
# ============================================================================

_ALL_METRICS = [
    HTTP_REQUEST_COUNT,
    HTTP_REQUEST_LATENCY,
    QUERY_LATENCY,
    QUERY_COUNT,
    QUERY_CONFIDENCE,
    GROUNDED_ANSWERS,
    DOCUMENTS_LOADED,
    DOCUMENT_UPLOADS,
    ACTIVE_USERS,
]


def metrics_endpoint() -> PlainTextResponse:
    """Return Prometheus-compatible text exposition."""
    body = "\n\n".join(m.collect() for m in _ALL_METRICS) + "\n"
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")


# ============================================================================
# Convenience helpers called from main.py
# ============================================================================

def track_query_latency(seconds: float):
    QUERY_LATENCY.observe(seconds)
    QUERY_COUNT.inc()


def track_query_accuracy(confidence: float, is_grounded: bool):
    QUERY_CONFIDENCE.observe(confidence)
    GROUNDED_ANSWERS.inc(grounded=str(is_grounded).lower())


def track_document_upload(filename: str):
    DOCUMENT_UPLOADS.inc()


# ============================================================================
# ASGI middleware
# ============================================================================

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        HTTP_REQUEST_LATENCY.observe(duration)
        HTTP_REQUEST_COUNT.inc(
            method=request.method,
            endpoint=request.url.path,
            status=str(response.status_code),
        )
        return response
