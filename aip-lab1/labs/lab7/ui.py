#!/usr/bin/env python3
"""Lab 7 — Streamlit front end.

    streamlit run labs/lab7/ui.py

Requires the service to be running:
    uvicorn labs.lab7.service:app --port 8000

Citations are rendered as expandable expanders showing the full source excerpt.
Grounding is only useful if the user can check it — a citation nobody can open
is decoration.
"""
from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="Aurora Policy Assistant", page_icon="🏥", layout="centered")

API = st.sidebar.text_input("Service URL", "http://localhost:8000")

# Sidebar: service health
if st.sidebar.button("Check service health"):
    try:
        h = requests.get(f"{API}/health", timeout=5)
        if h.ok:
            st.sidebar.success("Service is up ✓")
            hdata = h.json()
            st.sidebar.json({
                "uptime": f"{hdata.get('uptime_s', 0):.0f}s",
                "chunks": hdata.get("index_size_chunks", "?"),
                "profile": hdata.get("model_profile", "?"),
                "cache_entries": hdata.get("cache", {}),
            })
        else:
            st.sidebar.error(f"Service returned {h.status_code}")
    except requests.RequestException as exc:
        st.sidebar.error(f"Unreachable: {exc}")

# Sidebar: live metrics
if st.sidebar.button("Show metrics"):
    try:
        m = requests.get(f"{API}/metrics", timeout=5)
        if m.ok:
            mdata = m.json()
            st.sidebar.json({
                "total_requests": mdata.get("total_requests", 0),
                "cache_hit_rate": f"{mdata.get('cache_hit_rate', 0):.1%}",
                "cost_total": f"${mdata.get('cost_usd_total', 0):.5f}",
                "cost_per_query": f"${mdata.get('cost_per_query_usd', 0):.5f}",
                "p50_ms": mdata.get("p50_latency_ms", 0),
                "p95_cached_ms": mdata.get("p95_latency_cached_ms", 0),
                "p95_uncached_ms": mdata.get("p95_latency_uncached_ms", 0),
                "refusal_rate": f"{mdata.get('refusal_rate', 0):.1%}",
                "error_rate": f"{mdata.get('error_rate', 0):.1%}",
            })
    except requests.RequestException as exc:
        st.sidebar.error(f"Unreachable: {exc}")

st.title("🏥 Aurora Policy Assistant")
st.caption(
    "Answers come only from Aurora's policy documents. "
    "Every claim is cited. When the documents do not cover a question, "
    "the assistant says so instead of guessing."
)

# Example questions
with st.expander("📋 Example questions"):
    st.markdown("""
- How long do I have to file a reimbursement claim?
- What is the waiting period for pre-existing conditions?
- Is knee replacement covered immediately?
- Which plans have no co-payment?
- What is the claim settlement timeline?
""")

q = st.text_input(
    "Ask a question",
    placeholder="How long do I have to file a reimbursement claim?",
)

col1, col2, col3 = st.columns([2, 2, 6])
ask_btn = col1.button("Ask", type="primary")
clear_btn = col2.button("Clear")

if clear_btn:
    st.rerun()

stream_mode = st.sidebar.toggle("Stream the answer (/ask/stream)", value=False)


def _stream(question: str) -> None:
    """B3: prose streams in; citations arrive, validated, in the final event."""
    import json as _json

    box = st.empty()
    text, final = "", None
    try:
        with requests.post(f"{API}/ask/stream", json={"question": question},
                           stream=True, timeout=120) as r:
            if r.status_code != 200:
                st.error(f"{r.status_code}: {r.text[:300]}")
                return
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                ev = _json.loads(payload)
                if "token" in ev:
                    text += ev["token"]
                    box.markdown(text + " ▌")
                elif "error" in ev:
                    st.error(f"{ev['error']}: {ev.get('detail', '')}")
                    return
                else:
                    final = ev
    except requests.RequestException as exc:
        st.error(f"Service unreachable: {exc}")
        return

    box.markdown(text)
    if not final:
        return
    if final.get("refused"):
        st.warning("⚠️ The documents do not cover this question.")
    elif not final.get("citations_valid", True):
        st.error("⚠️ This answer cited sources that do not exist "
                 f"({final.get('invalid_citations')}). Do not rely on it.")
    if final.get("citations"):
        st.subheader("📚 Sources")
        for c in final["citations"]:
            with st.expander(f"[{c['index']}] {c['doc_id']}"):
                st.markdown(f"> {c['excerpt']}")
    st.divider()
    cols = st.columns(4)
    cols[0].metric("time to first token", f"{final.get('ttft_ms', 0):.0f} ms")
    cols[1].metric("total", f"{final.get('latency_ms', 0):.0f} ms")
    cols[2].metric("cost", f"${final.get('cost_usd', 0):.5f}")
    cols[3].metric("sources", len(final.get("citations", [])))
    st.caption(f"trace id: `{final.get('trace_id', '')}`")


if ask_btn and q and stream_mode:
    _stream(q)
    st.stop()

if ask_btn and q:
    with st.spinner("thinking…"):
        try:
            r = requests.post(
                f"{API}/ask",
                json={"question": q},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
        except requests.HTTPError as exc:
            code = exc.response.status_code
            if code == 429:
                st.error("💰 Budget exhausted — try again later.")
            elif code == 503:
                st.error("🔌 Service temporarily unavailable — try again in 30 s.")
            else:
                st.error(f"{code}: {exc.response.text[:300]}")
            st.stop()
        except requests.RequestException as exc:
            st.error(f"Service unreachable: {exc}")
            st.stop()

    # ── Answer ──────────────────────────────────────────────────────────────
    if data.get("refused"):
        st.warning("⚠️ " + data["answer"])
    else:
        st.markdown(data["answer"])

    # ── Citations — A4: expandable source text ───────────────────────────────
    citations = data.get("citations", [])
    if citations and not data.get("refused"):
        st.subheader("📚 Sources")
        for c in citations:
            with st.expander(f"[{c['index']}] {c['doc_id']}"):
                st.markdown(
                    f"> {c['excerpt']}",
                    help="This is the excerpt from the policy document that supports the answer above.",
                )

    # ── Metrics row ──────────────────────────────────────────────────────────
    st.divider()
    cols = st.columns(5)
    cols[0].metric("latency", f"{data.get('latency_ms', 0):.0f} ms")
    cols[1].metric("cost", f"${data.get('cost_usd', 0):.5f}")
    cols[2].metric("cached", "✅ yes" if data.get("cached") else "❌ no")
    cols[3].metric("sources", len(citations))
    cols[4].metric("refused", "yes" if data.get("refused") else "no")
    st.caption(f"trace id: `{data.get('trace_id', '')}`")

    # ── Thumbs-down feedback (stretch) ──────────────────────────────────────
    st.divider()
    if st.button("👎 Report a bad answer"):
        review_dir = __import__("pathlib").Path(".aip_review_queue")
        review_dir.mkdir(exist_ok=True)
        import json as _json
        import time as _time
        entry = {
            "ts": _time.time(),
            "question": q,
            "answer": data.get("answer"),
            "trace_id": data.get("trace_id"),
            "citations": citations,
        }
        fname = review_dir / f"{int(_time.time())}.json"
        fname.write_text(_json.dumps(entry, indent=2), encoding="utf-8")
        st.success("Feedback recorded — this case has been added to the review queue.")
