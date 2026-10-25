#!/usr/bin/env python3
"""Lab 7 service smoke test -- runs with the stand-in model, no API key.

    PYTHONPATH=tests/fake_provider python tests/lab7_smoke.py

Run by tests/test_lab7.py in a subprocess, so its environment (an isolated
cache, the fake provider) cannot leak into other tests. Checks the A3 status
codes, both cache layers, real streaming with held citations, /metrics, and
the refusal alert. It checks *behaviour*, not answer quality: the stand-in
model's answers are synthetic.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="lab7_smoke_"))
os.environ.update(AIP_CACHE_DIR=str(TMP / "cache"), AIP_TRACE_DIR=str(TMP / "traces"),
                  AIP_PROFILE="gemini", AIP_OFFLINE="0", AIP_BUDGET_USD="1",
                  FAKE_LLM_TOKEN_DELAY_S="0")
sys.path[:0] = [str(ROOT / "tests/fake_provider"), str(ROOT)]

import litellm  # noqa: E402

assert "fake_provider" in litellm.__file__, "the stand-in provider must shadow litellm"

from fastapi.testclient import TestClient  # noqa: E402

from labs.lab7 import service  # noqa: E402

client = TestClient(service.app)
ANSWERABLE = "How many days do I have to submit a reimbursement claim after discharge?"
OFF_TOPIC = "Does Aurora cover cosmetic dentistry on Mars?"


def check(name: str, cond: bool, detail: object = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + str(detail) if not cond else ''}")
    if not cond:
        sys.exit(1)


# health
h = client.get("/health").json()
check("health: pipeline built at startup", h["status"] == "ok" and h["index_size_chunks"] > 0, h)

# A1: response contract
r = client.post("/ask", json={"question": ANSWERABLE})
d = r.json()
check("A1: 200 with cost and trace id in body", r.status_code == 200
      and {"answer", "citations", "cost_usd", "trace_id", "latency_ms", "cached"} <= d.keys(), d)
check("A1: uncached answer is cited and costs money", not d["cached"] and d["cost_usd"] > 0
      and len(d["citations"]) > 0, d)

# refusal
r = client.post("/ask", json={"question": OFF_TOPIC}).json()
check("refusal: off-topic question refused with no citations", r["refused"] and r["citations"] == [], r)

# B1: caches
e = client.post("/ask", json={"question": "  how many DAYS do I have to submit a "
                                          "reimbursement claim after discharge? "}).json()
check("B1: exact cache hit after normalisation", e["cache_layer"] == "exact" and e["cost_usd"] == 0
      and e["source_trace_id"] == d["trace_id"] and e["trace_id"] != d["trace_id"], e)
s = client.post("/ask", json={"question": "How many days do I have to submit my "
                                          "reimbursement claim after discharge?"}).json()
check("B1: semantic cache hit on near-identical wording", s["cache_layer"] == "semantic", s)

# A3: 422
check("A3: malformed request -> 422", client.post("/ask", json={"question": "hi"}).status_code == 422)

# B2/B3: streaming
with client.stream("POST", "/ask/stream", json={"question": "What is the room rent limit "
                                                            "on the Silver plan?"}) as resp:
    events = [json.loads(line[6:]) for line in resp.iter_lines()
              if line.startswith("data: ") and line[6:] != "[DONE]"]
tokens = [e for e in events if "token" in e]
final = events[-1]
check("B2: streams several token events", len(tokens) > 3, len(tokens))
check("B3: citations held to the final event, validated", "citations" in final
      and "citations_valid" in final and all("citations" not in t for t in tokens), final)
check("B2: TTFT reported and below total latency", 0 < final["ttft_ms"] <= final["latency_ms"], final)

# B3: invalid citation is flagged in the final event
os.environ["FAKE_LLM_BAD_CITATION"] = "1"
with client.stream("POST", "/ask/stream", json={"question": "Which hospitals are part of the "
                                                            "cashless network hospitals list?"}) as resp:
    events = [json.loads(line[6:]) for line in resp.iter_lines()
              if line.startswith("data: ") and line[6:] != "[DONE]"]
os.environ.pop("FAKE_LLM_BAD_CITATION")
final = events[-1]
check("B3: hallucinated citation flagged for the UI",
      final["refused"] is False and final["citations_valid"] is False
      and final["invalid_citations"] == [9], final)

# A3: 503 on provider outage (fresh question so nothing is cached)
os.environ["FAKE_LLM_OUTAGE"] = "1"
import aip.llm  # noqa: E402

aip.llm.settings.max_retries = 1          # don't wait through backoff in a test
r = client.post("/ask", json={"question": "Is knee replacement covered immediately?"})
os.environ.pop("FAKE_LLM_OUTAGE")
check("A3: provider outage -> 503 with Retry-After, no stack trace",
      r.status_code == 503 and r.headers.get("retry-after") == "30"
      and "Traceback" not in r.text, (r.status_code, r.text))

# A3: 500 for a genuine bug, without a stack trace
orig = service.pipeline


def broken():
    raise KeyError("simulated bug")


service.pipeline = broken
r = client.post("/ask", json={"question": "What does the glossary say about co-payment?"})
service.pipeline = orig
check("A3: genuine bug -> 500, generic message", r.status_code == 500
      and r.json()["detail"] == "internal error" and "KeyError" not in r.text, r.text)

# A3: 429 when the budget is gone
from aip.cost import global_budget  # noqa: E402

global_budget().limit_usd = global_budget().spent_usd + 1e-9
r = client.post("/ask", json={"question": "Which plans have no co-payment at all?"})
check("A3: budget exhausted -> 429", r.status_code == 429, r.text)
global_budget().limit_usd = 1.0

# C2: metrics
m = client.get("/metrics").json()
needed = {"cost_usd_total", "cost_per_query_usd", "cache_hit_rate", "p50_latency_ms",
          "p95_latency_ms", "p99_latency_ms", "p95_latency_cached_ms", "p95_latency_uncached_ms",
          "error_rate", "errors_by_type", "refusal_rate", "alert_refusal_rate_doubled"}
check("C2: /metrics has every required field", needed <= m.keys(), needed - m.keys())
check("C2: errors counted by type", {"provider_outage", "internal_error", "budget_exceeded"}
      <= m["errors_by_type"].keys(), m["errors_by_type"])

# C4: alert logic
quiet = [{"refused": i % 5 == 0} for i in range(40)]
spike = quiet[:20] + [{"refused": True}] * 20
check("C4: alert quiet at a steady refusal rate", not service.refusal_alert(quiet)["firing"])
check("C4: alert fires when refusals double", service.refusal_alert(spike)["firing"])

print("\nAll Lab 7 smoke checks passed.")
