#!/usr/bin/env python3
"""Lab 7 -- the regression gate. Exits non-zero when a threshold is breached.

    python labs/lab7/gate.py                      # live: calls the model, fills the cache
    AIP_OFFLINE=1 python labs/lab7/gate.py        # CI: replays the committed cache, $0

How it works
------------
The gate runs the *shipped* pipeline (labs/lab7/pipeline.py -- the same object
the service builds) over every golden question, then judges the answers.

With AIP_OFFLINE=1 every model call is served from .aip_cache/calls.sqlite3.
A call is a cache hit only if its request is byte-identical to one recorded
earlier, so the cache doubles as a tripwire: change FINAL_K, the chunking,
the prompt or the corpus, and the prompts change, the cache misses, and the
gate goes red. That is why D3 works offline.

Metrics (all measured; nothing is hardcoded)
--------------------------------------------
  correctness         LLM judge vs gold answer, 0/1/2 scaled to 0..1
  faithfulness        LLM judge: is every claim supported by the context (0/1)
  citation_validity   every [n] indexes a real source (refusals count as valid)
  refusal_recall      unanswerable questions that were refused
  refusal_precision   refusals that were on unanswerable questions
  hit_rate_at_5       a relevant doc is among the chunks the generator saw
  cost_per_query_usd  recorded cost of the generation call (uncached price)
  p95_latency_ms      recorded generation latency + measured retrieval time
  cache_coverage      share of golden questions reproducible from the cache
                      (only < 1.0 offline; a miss means the pipeline changed
                      or the cache was not warmed for this configuration)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GOLDEN_PATH = ROOT / "data/eval/rag_golden.jsonl"
SETUP_ERROR = 2


def _load_golden() -> list[dict]:
    with GOLDEN_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _check_cache(offline: bool) -> str | None:
    """Return an error message if the offline cache is unusable, else None."""
    from aip.config import settings

    db = settings.cache_dir / "calls.sqlite3"
    if not offline:
        return None
    if not db.exists() or db.stat().st_size == 0:
        return (f"AIP_OFFLINE=1 but {db} is missing or empty.\n"
                "Run the gate once online to warm it, then commit it:\n"
                "  python labs/lab7/gate.py\n"
                "  git add -f .aip_cache/calls.sqlite3 && git commit -m 'Warm eval cache'")
    with contextlib.closing(sqlite3.connect(db)) as conn:
        has = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='calls'"
        ).fetchone()
        n = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0] if has else 0
    if not n:
        return f"{db} has no recorded calls. Warm it online first (see above)."
    return None


class _UsageTap:
    """Records the *original* cost and latency of every model call.

    On a cache hit aip.llm reports cost 0 and latency 0 (correct for billing),
    but the cached response still carries what the call cost when it was
    recorded. That is the number the gate needs: the uncached price.
    """

    def __init__(self):
        self.calls: list[dict] = []

    @contextlib.contextmanager
    def install(self):
        import aip.llm as llm

        original = llm.raw_call

        def tapped(*args, **kwargs):
            t0 = time.perf_counter()
            res = original(*args, **kwargs)
            u = res.get("usage", {})
            self.calls.append({
                "cost_usd": float(u.get("cost_usd") or 0.0),
                "recorded_latency_ms": float(u.get("latency_ms") or 0.0),
                "wall_ms": (time.perf_counter() - t0) * 1000,
            })
            return res

        llm.raw_call = tapped
        try:
            yield self
        finally:
            llm.raw_call = original


def measure(limit: int | None = None, verbose: bool = False) -> tuple[dict, list[dict]]:
    from aip.cache import CacheMiss
    from aip.evals import llm_judge, retrieval_metrics
    from aip.guards import enforce_citations
    from aip.retrieval import format_context
    from labs.lab4.evaluate import CORRECTNESS_RUBRIC, FAITHFULNESS_RUBRIC
    from labs.lab7.pipeline import build_pipeline

    rows = _load_golden()[:limit] if limit else _load_golden()
    pipe = build_pipeline()        # raises CacheMiss offline if embeddings missing
    tap = _UsageTap()

    details: list[dict] = []
    with tap.install():
        for row in rows:
            d = {"id": row["id"], "kind": row["kind"], "status": "ok"}
            try:
                # Retrieval is local (embeddings come from the cache), so it is
                # measured for every question -- even when generation is not
                # reproducible. This is what shows the real damage of a D3 break.
                d["final_docs"] = [h.doc_id for h in
                                   pipe.rerank(row["question"], pipe.retrieve(row["question"]))]
            except CacheMiss:
                d["status"] = "cache_miss"
                details.append(d)
                continue
            try:
                n0 = len(tap.calls)
                t0 = time.perf_counter()
                result = pipe.answer(row["question"])
                wall_ms = (time.perf_counter() - t0) * 1000
                gen = tap.calls[n0:]
                d.update(
                    refused=result.refused,
                    cost_usd=sum(c["cost_usd"] for c in gen),
                    # end-to-end as if uncached: our retrieval time + recorded model time
                    latency_ms=wall_ms - sum(c["wall_ms"] for c in gen)
                    + sum(c["recorded_latency_ms"] for c in gen),
                )
                ok, _invalid = enforce_citations(result.answer, len(result.hits))
                d["citation_valid"] = 1.0 if result.refused else float(ok)

                if row.get("gold_answer"):
                    v = llm_judge(CORRECTNESS_RUBRIC.format(
                        question=row["question"], reference=row["gold_answer"],
                        candidate=result.answer), tier="LARGE")
                    d["correctness"] = min(1.0, float(v.get("score", 0)) / 2.0)
                if not result.refused:
                    v = llm_judge(FAITHFULNESS_RUBRIC.format(
                        context=format_context(result.hits)[:3000],
                        answer=result.answer), tier="LARGE")
                    d["faithfulness"] = float(v.get("score", 0))
            except CacheMiss:
                d["status"] = "cache_miss"
            details.append(d)
            if verbose:
                print(f"  {row['id']}: {d['status']}"
                      f"{' refused' if d.get('refused') else ''}", file=sys.stderr)

    done = [d for d in details if d["status"] == "ok"]
    by_id = {r["id"]: r for r in rows}

    def mean(xs):
        xs = list(xs)
        return round(statistics.fmean(xs), 4) if xs else None   # None = not measured

    hit = [retrieval_metrics(d["final_docs"], by_id[d["id"]]["relevant_docs"], ks=(5,))["hit_rate@5"]
           for d in details if "final_docs" in d and by_id[d["id"]].get("relevant_docs")]
    unans = [d for d in done if d["kind"] == "unanswerable"]
    refusals = [d for d in done if d["refused"]]
    lats = sorted(d["latency_ms"] for d in done)
    p95 = lats[min(len(lats) - 1, int(round(0.95 * (len(lats) - 1))))] if lats else None

    metrics = {
        "correctness": mean(d["correctness"] for d in done if "correctness" in d),
        "faithfulness": mean(d["faithfulness"] for d in done if "faithfulness" in d),
        "citation_validity": mean(d["citation_valid"] for d in done),
        "refusal_recall": round(sum(d["refused"] for d in unans) / len(unans), 4) if unans else None,
        "refusal_precision": (round(sum(1 for d in refusals if d["kind"] == "unanswerable")
                                    / len(refusals), 4) if refusals else 1.0) if done else None,
        "hit_rate_at_5": mean(hit),
        "cost_per_query_usd": mean(d["cost_usd"] for d in done),
        "p95_latency_ms": round(p95, 1) if lats else None,
        "cache_coverage": round(len(done) / len(rows), 4) if rows else 0.0,
    }
    return metrics, details


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="labs/lab7/thresholds.yml")
    ap.add_argument("--limit", type=int, default=None, help="first N golden questions only")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    from aip.cache import CacheMiss
    from aip.config import settings
    from labs.lab7.pipeline import config_dict

    thresholds = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))

    err = _check_cache(settings.offline)
    if err:
        print(f"GATE SETUP ERROR:\n{err}", file=sys.stderr)
        return SETUP_ERROR

    mode = "offline replay" if settings.offline else "live"
    print(f"Running golden set ({mode}) with pipeline {config_dict()} ...")
    try:
        metrics, details = measure(args.limit, args.verbose)
    except CacheMiss as exc:
        # The index itself could not be rebuilt from the cache: chunking or
        # the corpus changed. That is a real regression signal, not a crash.
        print(f"\nGATE FAILED: pipeline could not be rebuilt from the cache.\n{exc}")
        return 1

    failures = []
    width = max(len(k) for k in thresholds)
    print(f"\n{'metric':<{width}}  {'value':>10}  {'gate':>10}  status")
    print("-" * (width + 36))
    for name, rule in thresholds.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: not measured (no reproducible answers)")
            print(f"{name:<{width}}  {'-':>10}  {'':>10}  NOT MEASURED")
            continue
        ok, gate = True, ""
        if "min" in rule:
            gate, ok = f">= {rule['min']}", value >= rule["min"]
        if "max" in rule and ok:
            gate, ok = f"<= {rule['max']}", value <= rule["max"]
        if not ok:
            failures.append(f"{name}: {value} violates {gate}")
        print(f"{name:<{width}}  {value:>10.4f}  {gate:>10}  {'ok' if ok else 'FAIL'}")

    misses = [d["id"] for d in details if d["status"] == "cache_miss"]
    if misses:
        print(f"\n{len(misses)} question(s) not reproducible from the cache: "
              f"{', '.join(misses[:10])}{' ...' if len(misses) > 10 else ''}")

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    report = {"mode": mode, "pipeline": config_dict(), "metrics": metrics,
              "thresholds": thresholds, "failures": failures, "passed": not failures,
              "n_questions": len(details), "cache_misses": misses}
    (report_dir / "gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (report_dir / "gate_details.jsonl").open("w", encoding="utf-8") as fh:
        for d in details:
            fh.write(json.dumps(d) + "\n")
    print("\nReport saved to reports/gate_report.json (per-question: gate_details.jsonl)")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
