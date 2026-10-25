#!/usr/bin/env python3
"""Lab 7 — the observability dashboard, read from local traces.

    streamlit run labs/lab7/dashboard.py

`aip.tracing` writes one JSONL file per run to .aip_traces/. This page reads
them back. It is a teaching-scale stand-in for Langfuse / LangSmith / Phoenix;
the concept -- structured spans with a run id and a parent id -- is identical.

C4 ALERT: The one alert condition implemented here is:
    Refusal rate doubling (from its baseline).

Rationale: A broken or stale index produces no errors, no latency spike, and no
cost increase — it just quietly stops finding things. A well-built RAG system
responds by declining. The refusal rate is therefore the canary for a silent
data failure, and watching it is free.

When it fires: investigate `retriever.search` spans first. If hit-rate metrics
have dropped, the index needs rebuilding. If hits are present but answers are
still refusals, the corpus may have changed in a way that invalidates the
generation prompt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.config import settings  # noqa: E402

st.set_page_config(page_title="Aurora Assistant — Ops", layout="wide")
st.title("Aurora Policy Assistant — Operations Dashboard")

# ---------------------------------------------------------------------------
# Load traces
# ---------------------------------------------------------------------------
runs = sorted(settings.trace_dir.glob("*.jsonl"), reverse=True)
if not runs:
    st.info(f"No traces yet in {settings.trace_dir}. Run some queries first.")
    st.stop()

chosen = st.sidebar.multiselect(
    "runs",
    [p.stem for p in runs],
    default=[runs[0].stem],
)
rows = [json.loads(l) for p in runs if p.stem in chosen
        for l in p.open(encoding="utf-8") if l.strip()]
if not rows:
    st.stop()

df = pd.DataFrame(rows)
df["ts"] = pd.to_datetime(df["ts"], unit="s")

# ---------------------------------------------------------------------------
# Top-level KPIs
# ---------------------------------------------------------------------------
c = st.columns(6)
c[0].metric("spans", len(df))
c[1].metric("total cost", f"${df.get('cost_usd', pd.Series([0])).fillna(0).sum():.4f}")

llm = df[df["name"] == "llm.call"] if "name" in df.columns else pd.DataFrame()
c[2].metric("model calls", len(llm))
if len(llm) and "cached" in llm.columns:
    cache_rate = llm["cached"].fillna(False).mean()
    c[3].metric("cache hit rate", f"{cache_rate:.0%}")
else:
    c[3].metric("cache hit rate", "n/a")

n_errors = int((df.get("status", pd.Series([])) == "error").sum()) if "status" in df.columns else 0
c[4].metric("errors", n_errors)

# Refusal rate from ask spans (non-streaming and streaming requests)
ask_spans = (df[df["name"].isin(["http.ask", "http.ask.stream"])]
             if "name" in df.columns else pd.DataFrame())
if len(ask_spans) > 0 and "refused" in ask_spans.columns:
    refusal_rate = ask_spans["refused"].fillna(False).mean()
    c[5].metric("refusal rate", f"{refusal_rate:.0%}")
else:
    c[5].metric("refusal rate", "n/a")

# ---------------------------------------------------------------------------
# C3 — Latency by stage (p50/p95 per span name)
# ---------------------------------------------------------------------------
st.subheader("⏱ Latency by Stage")
st.caption("This table answers: *which stage should I optimise first?*")

if "duration_ms" in df.columns and "name" in df.columns:
    stage = (df.groupby("name")["duration_ms"]
             .agg(n="count",
                  p50="median",
                  p95=lambda s: s.quantile(0.95),
                  p99=lambda s: s.quantile(0.99),
                  total="sum")
             .round(1)
             .sort_values("total", ascending=False))
    st.dataframe(stage, use_container_width=True)

    # Bar chart of p95 by stage
    st.bar_chart(stage["p95"])
else:
    st.info("No duration_ms data yet.")

# ---------------------------------------------------------------------------
# Latency over time (rolling p95)
# ---------------------------------------------------------------------------
st.subheader("📈 End-to-End Latency Over Time")
if len(ask_spans) > 0 and "duration_ms" in ask_spans.columns:
    latency_series = ask_spans.set_index("ts")["duration_ms"].sort_index()
    if len(latency_series) >= 5:
        rolling_p95 = latency_series.rolling("5min").quantile(0.95)
        combined = pd.DataFrame({"latency_ms": latency_series, "p95_5min": rolling_p95})
        st.line_chart(combined)
    else:
        st.line_chart(latency_series)
else:
    st.info("No http.ask spans yet.")

# ---------------------------------------------------------------------------
# C2 — Cumulative cost over time
# ---------------------------------------------------------------------------
st.subheader("💰 Cumulative Cost Over Time")
if "cost_usd" in df.columns:
    cum = (df.sort_values("ts")
           .assign(cum=lambda d: d["cost_usd"].fillna(0).cumsum()))
    st.line_chart(cum.set_index("ts")["cum"])
else:
    st.info("No cost_usd data yet.")

# ---------------------------------------------------------------------------
# Cache hit rate over time
# ---------------------------------------------------------------------------
st.subheader("🎯 Cache Hit Rate")
if len(ask_spans) > 0 and "cache_hit" in ask_spans.columns:
    cache_col = ask_spans.set_index("ts")["cache_hit"].sort_index()
    is_hit = (cache_col != "none").astype(int)
    rolling_hit = is_hit.rolling("5min").mean()
    st.line_chart(pd.DataFrame({
        "is_hit (1=hit)": is_hit,
        "rolling_5min_avg": rolling_hit,
    }))
elif len(llm) > 0 and "cached" in llm.columns:
    cache_series = llm.set_index("ts")["cached"].fillna(False).astype(int).sort_index()
    st.line_chart(cache_series)
else:
    st.info("No cache_hit data yet in traces — check service.py is writing spans.")

# ---------------------------------------------------------------------------
# Error log
# ---------------------------------------------------------------------------
st.subheader("❌ Errors")
if "status" in df.columns:
    errs = df[df["status"] == "error"]
    cols_to_show = [col for col in ["ts", "name", "error"] if col in errs.columns]
    st.dataframe(errs[cols_to_show] if len(errs) and cols_to_show else pd.DataFrame(),
                 use_container_width=True)
else:
    st.info("No status column in traces.")

# ---------------------------------------------------------------------------
# C4 — Alert condition: refusal rate doubling
# ---------------------------------------------------------------------------
st.subheader("🚨 Alert: Refusal Rate Monitor")
st.markdown("""
**Alert condition:** the last 5 minutes' refusal rate is at least double the preceding 30-minute baseline.

**Why this one?** A broken index produces no errors, no latency spike, and no cost change.
It just quietly stops finding relevant chunks — and a correctly-built RAG system responds
by *declining to answer*. The refusal rate is therefore the first signal of a silent data failure.

**When it fires:** Check `rag.retrieve` spans for hit count. If `n_hits` is dropping,
the index needs rebuilding. If hits are present but answers are still refusals, the corpus
content may have drifted out of scope relative to the queries.
""")

RECENT = pd.Timedelta(minutes=5)     # the window we are judging
BASELINE = pd.Timedelta(minutes=30)  # the rolling window it is compared with

if len(ask_spans) > 0 and "refused" in ask_spans.columns:
    ref = ask_spans.set_index("ts")["refused"].fillna(False).astype(int).sort_index()
    now = ref.index.max()
    recent = ref[ref.index > now - RECENT]
    base = ref[(ref.index > now - RECENT - BASELINE) & (ref.index <= now - RECENT)]
    if len(recent) >= 5 and len(base) >= 10:
        b, r = base.mean(), recent.mean()
        firing = (r >= 2 * b and r > 0) if b > 0 else r >= 0.25
        if firing:
            st.error(f"🔴 ALERT: refusal rate doubled — 30-min baseline {b:.0%} "
                     f"→ last 5 min {r:.0%}. Check rag.retrieve n_hits, then the corpus.")
        else:
            st.success(f"✅ Refusal rate normal — 30-min baseline {b:.0%}, last 5 min {r:.0%}.")
    else:
        st.info(f"Not enough data for the alert yet: {len(base)} requests in the 30-min "
                f"baseline (need 10) and {len(recent)} in the last 5 min (need 5). "
                "Showing the raw series below.")
    st.line_chart(pd.DataFrame({
        "refused": ref,
        "rolling_30min": ref.rolling("30min").mean(),
    }))
else:
    st.info("No refusal data in traces yet.")

# ---------------------------------------------------------------------------
# Span inspector
# ---------------------------------------------------------------------------
with st.expander("🔍 Raw span inspector"):
    st.dataframe(df, use_container_width=True)
    st.caption("Filter by 'name' column to see stage-level detail.")
