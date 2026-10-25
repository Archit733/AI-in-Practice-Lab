#!/usr/bin/env python3
"""Lab 7 -- the service.

    uvicorn labs.lab7.service:app --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"How long do I have to file a claim?"}' | jq

Pipeline: built once at startup from labs/lab7/pipeline.py -- the same
definition the regression gate tests.

Two cache layers (B1):
  1. Exact response cache -- SHA-256 of (pipeline config + normalised question).
     The config is part of the key, so changing FINAL_K or the chunking cannot
     serve answers produced by the old pipeline.
  2. Semantic cache -- cosine similarity of question embeddings, threshold 0.95.
     Returns the *closest* stored question, and logs which one matched and how
     similar it was, so a wrong hit is diagnosable from the trace.
     At ~0.93 questions differing only by plan name (Gold vs Silver) collide;
     labs/lab7/sem_sweep.py measures this.

Streaming (B2/B3): POST /ask/stream streams real model tokens as they are
generated. Strategy chosen: stream the prose, hold the citations to the end.
Citations are validated once generation finishes and arrive in the final
event with `citations_valid` / `invalid_citations`, which the UI acts on.

Errors (A3): 422 malformed request (Pydantic) - 429 budget exhausted -
503 + Retry-After for model outages or an offline cache miss - 500 with a
trace id (never a stack trace) for genuine bugs.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, cost, tracing  # noqa: E402
from aip.cache import CacheMiss  # noqa: E402
from aip.cost import BudgetExceeded, global_budget  # noqa: E402
from aip.embed import embed  # noqa: E402
from labs.lab7 import pipeline as shipped  # noqa: E402

app = FastAPI(title="Aurora Policy Assistant", version="1.1")

_PIPELINE = None
_STARTUP_ERROR: str | None = None
_STARTED = time.time()
_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Semantic cache (in-memory): (embedding, question, response)
# ---------------------------------------------------------------------------
_SEM_THRESHOLD = 0.95   # see labs/lab7/sem_sweep.py for the measurement
_SEM_CACHE: list[tuple[np.ndarray, str, dict]] = []

# ---------------------------------------------------------------------------
# Request ledger for /metrics and the refusal alert
# ---------------------------------------------------------------------------
_REQUESTS: deque[dict] = deque(maxlen=10_000)  # {ts, latency_ms, layer, refused, cost_usd}
_ERRORS: deque[dict] = deque(maxlen=10_000)
_TOTAL = 0

PROVIDER_OUTAGE_MARKERS = ("503", "502", "overloaded", "serviceunavailable", "unavailable",
                           "rate_limit", "ratelimit", "429", "timeout", "apiconnection")


def pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = shipped.build_pipeline()
    return _PIPELINE


try:  # build at startup, not on the first request
    pipeline()
except Exception as exc:  # noqa: BLE001 -- surfaced in /health and as 503s
    _STARTUP_ERROR = f"{type(exc).__name__}: {str(exc)[:300]}"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    # top_k / mode are accepted for contract compatibility. The shipped
    # configuration is fixed in pipeline.py (FINAL_K) so the gate tests
    # exactly what is served; "tools" mode is not enabled in this deployment.
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    latency_ms: float
    cost_usd: float
    cached: bool
    cache_layer: str = "none"          # "exact" | "semantic" | "none"
    trace_id: str
    source_trace_id: str | None = None  # trace of the request that produced a cached answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm_question(q: str) -> str:
    return " ".join(q.lower().split())


def _exact_key(q: str) -> str:
    blob = json.dumps({"cfg": shipped.config_dict(), "q": _norm_question(q)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _sem_lookup(q_vec: np.ndarray) -> tuple[dict, str, float] | None:
    best: tuple[dict, str, float] | None = None
    for vec, question, resp in _SEM_CACHE:
        sim = float(q_vec @ vec)
        if sim >= _SEM_THRESHOLD and (best is None or sim > best[2]):
            best = (resp, question, sim)
    return best


def _citations(hits) -> list[dict]:
    return [{"index": i, "doc_id": h.doc_id, "excerpt": h.text[:400]}
            for i, h in enumerate(hits, start=1)]


def _record(latency_ms: float, layer: str, refused: bool, cost_usd: float) -> None:
    with _LOCK:
        _REQUESTS.append({"ts": time.time(), "latency_ms": latency_ms, "layer": layer,
                          "refused": bool(refused), "cost_usd": cost_usd})


def _record_error(kind: str, detail: str = "") -> None:
    with _LOCK:
        _ERRORS.append({"type": kind, "ts": time.time(), "detail": detail[:200]})


def _classify(exc: Exception) -> HTTPException:
    """Map an exception to the status code that tells the client what to do."""
    if isinstance(exc, BudgetExceeded):
        _record_error("budget_exceeded", str(exc))
        return HTTPException(429, detail="spend budget exhausted -- try later")
    if isinstance(exc, CacheMiss):
        _record_error("offline_cache_miss", str(exc))
        return HTTPException(503, detail="answer not available offline",
                             headers={"Retry-After": "30"})
    text = f"{type(exc).__name__} {exc}".lower()
    if any(m in text for m in PROVIDER_OUTAGE_MARKERS):
        _record_error("provider_outage", str(exc))
        return HTTPException(503, detail="upstream model unavailable -- try again shortly",
                             headers={"Retry-After": "30"})
    _record_error("internal_error", f"{type(exc).__name__}: {exc}")
    return HTTPException(500, detail="internal error")


def _ensure_ready() -> None:
    if _STARTUP_ERROR and _PIPELINE is None:
        try:
            pipeline()
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc) from exc


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """A1: answer + citations + latency + cost + trace id in the body."""
    global _TOTAL
    with _LOCK:
        _TOTAL += 1
    t0 = time.perf_counter()
    cost_before = global_budget().spent_usd

    try:
        _ensure_ready()
        with tracing.trace("http.ask", question=req.question[:120]) as span:
            trace_id = span["span_id"]

            # 1. exact cache
            key = _exact_key(req.question)
            hit = cache.get(key)
            if hit is not None:
                ms = (time.perf_counter() - t0) * 1000
                span.update(cache_hit="exact", refused=hit["refused"])
                _record(ms, "exact", hit["refused"], 0.0)
                return AskResponse(**{**hit, "latency_ms": round(ms, 1), "cost_usd": 0.0,
                                      "cached": True, "cache_layer": "exact",
                                      "trace_id": trace_id,
                                      "source_trace_id": hit.get("trace_id")})

            # 2. semantic cache
            q_vec = embed(req.question, input_type="query")
            sem = _sem_lookup(q_vec)
            if sem is not None:
                resp, matched_q, sim = sem
                ms = (time.perf_counter() - t0) * 1000
                span.update(cache_hit="semantic", refused=resp["refused"],
                            matched_question=matched_q[:120], similarity=round(sim, 4))
                _record(ms, "semantic", resp["refused"], 0.0)
                return AskResponse(**{**resp, "latency_ms": round(ms, 1), "cost_usd": 0.0,
                                      "cached": True, "cache_layer": "semantic",
                                      "trace_id": trace_id,
                                      "source_trace_id": resp.get("trace_id")})

            # 3. miss -> run the pipeline
            result = pipeline().answer(req.question)
            ms = (time.perf_counter() - t0) * 1000
            spent = max(0.0, global_budget().spent_usd - cost_before)
            span.update(cache_hit="none", refused=result.refused,
                        n_citations=len(result.hits), cost_usd=round(spent, 6))
            resp = {
                "answer": result.answer,
                "refused": result.refused,
                # A refusal is supported by nothing, so it cites nothing.
                "citations": [] if result.refused else _citations(result.hits),
                "latency_ms": round(ms, 1),
                "cost_usd": round(spent, 6),
                "cached": False,
                "cache_layer": "none",
                "trace_id": trace_id,
            }
            _record(ms, "none", result.refused, spent)
            cache.put(key, "response", {"question": req.question}, resp)
            _SEM_CACHE.append((q_vec, req.question, resp))
            return AskResponse(**resp)

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _classify(exc) from exc


# ---------------------------------------------------------------------------
# POST /ask/stream -- real token streaming, citations held to the end (B3)
# ---------------------------------------------------------------------------
def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """B2/B3. Retrieval happens before the response starts, so a budget or
    outage problem there still gets a proper status code. Once tokens flow the
    status is already 200, so later failures arrive as an `error` event."""
    from aip.config import resolve_model, settings
    from aip.guards import delimit_untrusted, enforce_citations
    from aip.llm import _normalise
    from aip.rag import REFUSAL
    from aip.retrieval import format_context

    global _TOTAL
    with _LOCK:
        _TOTAL += 1
    t0 = time.perf_counter()
    if global_budget().spent_usd >= global_budget().limit_usd:
        raise _classify(BudgetExceeded("budget exhausted"))
    try:
        _ensure_ready()
        pipe = pipeline()
        with tracing.trace("stream.retrieve", question=req.question[:120]):
            hits = pipe.rerank(req.question, pipe.retrieve(req.question))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _classify(exc) from exc

    # Same prompt and request shape as RagPipeline.generate -> same cache key,
    # so streamed and non-streamed answers share the model-call cache.
    context = delimit_untrusted(format_context(hits, max_chars=pipe.max_context_chars))
    prompt = f"{context}\n\nQuestion: {req.question}\n\nAnswer with citations:"
    model = resolve_model(pipe.tier)
    messages = _normalise(prompt, pipe.system)
    request = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 600,
               "response_format": None, "tools": None, "tool_choice": None}
    key = cache.make_key("chat", request)
    trace_id = uuid.uuid4().hex[:12]
    retrieve_ms = (time.perf_counter() - t0) * 1000

    def generate():
        parts: list[str] = []
        ttft_ms = None
        source = "model"
        usd = 0.0
        try:
            hit = cache.get(key)
            if hit is not None:
                source = "cache"
                words = hit["text"].split(" ")
                for i, w in enumerate(words):
                    tok = w + (" " if i < len(words) - 1 else "")
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    parts.append(tok)
                    yield _sse({"token": tok})
            elif settings.offline:
                raise CacheMiss("AIP_OFFLINE=1 and this streamed answer is not cached")
            else:
                from litellm import completion

                pt = ct = 0
                t_gen = time.perf_counter()
                for chunk in completion(model=model, messages=messages, temperature=0.0,
                                        max_tokens=600, stream=True,
                                        stream_options={"include_usage": True},
                                        timeout=settings.timeout_s):
                    if getattr(chunk, "usage", None):
                        pt = int(chunk.usage.prompt_tokens or 0)
                        ct = int(chunk.usage.completion_tokens or 0)
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        parts.append(delta)
                        yield _sse({"token": delta})
                gen_ms = (time.perf_counter() - t_gen) * 1000
                usd = cost.price_of(model, pt, ct)
                text = "".join(parts)
                cache.put(key, "chat", request, {
                    "text": text, "tool_calls": [], "finish_reason": "stop",
                    "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                              "cost_usd": usd, "latency_ms": gen_ms, "cached": False}})
                cost.record(cost.Usage(model, pt, ct, usd, gen_ms, cached=False, calls=1,
                                       priced=cost.is_priced(model)))

            text = "".join(parts).strip()
            refused = text.startswith(REFUSAL[:40])
            ok, invalid = enforce_citations(text, len(hits))
            total_ms = (time.perf_counter() - t0) * 1000
            _record(total_ms, "none", refused, usd)
            tracing.event("http.ask.stream", trace_id=trace_id, question=req.question[:120],
                          duration_ms=round(total_ms, 1), ttft_ms=round(ttft_ms or total_ms, 1),
                          retrieve_ms=round(retrieve_ms, 1), refused=refused,
                          citations_valid=ok or refused, source=source,
                          cost_usd=round(usd, 6), cache_hit="none")
            yield _sse({
                "citations": [] if refused else _citations(hits),
                "citations_valid": ok or refused,
                "invalid_citations": invalid,
                "refused": refused,
                "ttft_ms": round(ttft_ms or total_ms, 1),
                "latency_ms": round(total_ms, 1),
                "cost_usd": round(usd, 6),
                "trace_id": trace_id,
                "source": source,
            })
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001
            err = _classify(exc)
            tracing.event("http.ask.stream", trace_id=trace_id, status="error",
                          error=f"{type(exc).__name__}: {str(exc)[:200]}")
            yield _sse({"error": err.status_code, "detail": err.detail})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# GET /health and GET /metrics
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    from aip.config import settings

    n_chunks = 0
    if _PIPELINE is not None and hasattr(_PIPELINE.retriever, "retrievers"):
        n_chunks = len(_PIPELINE.retriever.retrievers[0].chunks)
    return {
        "status": "ok" if _PIPELINE is not None else "degraded",
        "startup_error": _STARTUP_ERROR if _PIPELINE is None else None,
        "uptime_s": round(time.time() - _STARTED, 1),
        "index_size_chunks": n_chunks,
        "pipeline": shipped.config_dict(),
        "model_profile": settings.profile,
        "models": settings.models,
        "offline": settings.offline,
        "cache": cache.stats(),
        "semantic_cache_size": len(_SEM_CACHE),
        "semantic_cache_threshold": _SEM_THRESHOLD,
    }


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 1)


def refusal_alert(reqs: list[dict], recent_n: int = 20, min_baseline: int = 20) -> dict:
    """C4: fire when the recent refusal rate is at least double the baseline.

    A broken index throws no errors, adds no latency and costs nothing -- the
    system just starts declining. Baseline = everything before the most recent
    `recent_n` requests (needs >= `min_baseline` of them).
    """
    if len(reqs) < recent_n + min_baseline:
        return {"firing": False, "reason": f"need {recent_n + min_baseline} requests, have {len(reqs)}"}
    base, recent = reqs[:-recent_n], reqs[-recent_n:]
    b = sum(r["refused"] for r in base) / len(base)
    r = sum(x["refused"] for x in recent) / len(recent)
    firing = r >= 2 * b and r > 0 if b > 0 else r >= 0.25
    return {"firing": firing, "baseline_refusal_rate": round(b, 4),
            "recent_refusal_rate": round(r, 4),
            "runbook": "check n_hits on rag.retrieve spans; 0 -> rebuild index; "
                       ">0 -> check recent corpus changes" if firing else None}


@app.get("/metrics")
def metrics() -> dict:
    """C2: cost, cost/query, cache hit rate, latency percentiles (cached and
    uncached quoted separately), error rate by type, refusal rate + alert."""
    b = global_budget()
    with _LOCK:
        reqs, errs, total = list(_REQUESTS), list(_ERRORS), _TOTAL
    answered = len(reqs)
    cached = [r for r in reqs if r["layer"] != "none"]
    uncached = [r for r in reqs if r["layer"] == "none"]
    lat = [r["latency_ms"] for r in reqs]
    by_type: dict[str, int] = {}
    for e in errs:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    return {
        "total_requests": total,
        "answered": answered,
        "cost_usd_total": round(b.spent_usd, 6),
        "cost_per_query_usd": round(b.spent_usd / answered, 6) if answered else 0.0,
        "cost_per_uncached_query_usd": round(sum(r["cost_usd"] for r in uncached) / len(uncached), 6)
        if uncached else 0.0,
        "exact_cache_hits": sum(r["layer"] == "exact" for r in reqs),
        "semantic_cache_hits": sum(r["layer"] == "semantic" for r in reqs),
        "cache_hit_rate": round(len(cached) / answered, 4) if answered else 0.0,
        "p50_latency_ms": _pct(lat, 50),
        "p95_latency_ms": _pct(lat, 95),
        "p99_latency_ms": _pct(lat, 99),
        "p95_latency_cached_ms": _pct([r["latency_ms"] for r in cached], 95),
        "p95_latency_uncached_ms": _pct([r["latency_ms"] for r in uncached], 95),
        "error_rate": round(len(errs) / total, 4) if total else 0.0,
        "errors_by_type": by_type,
        "refusal_rate": round(sum(r["refused"] for r in reqs) / answered, 4) if answered else 0.0,
        "alert_refusal_rate_doubled": refusal_alert(reqs),
        "model_calls": b.calls,
        "model_calls_cached": b.cached_calls,
        "unpriced_calls": b.unpriced_calls,
    }
