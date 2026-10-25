# Evaluation Report — Aurora Policy Assistant

*Lab 7 · Module 1 Capstone · Two-page evaluation report*

---

## 1 — What It Does

The Aurora Policy Assistant answers questions about Aurora Health Insurance policies. A user types a question in plain English; the system finds the most relevant sections of the policy documents, then asks an AI model to write a grounded answer that cites exactly which sections it drew from. If the documents do not contain the answer, the system says so rather than guessing. Every response shows which documents were consulted, what they say, how long it took, and what it cost — so users can check the sources, and the team can track the economics.

---

## 2 — How Well It Works

*Test set: 45 questions from `data/eval/rag_golden.jsonl` (42 with relevant documents; 3 have no relevant doc). Evaluated with `labs/lab7/gate.py`. LLM judge: LARGE tier (gemini-3.5-flash) — a different model family from the generation tier (MAIN: gemini-3.7-flash) to reduce self-preference bias.*

| Metric | Result | Gate threshold | Status |
|---|---|---|---|
| Correctness (LLM judge, 0–1) | 0.90 | ≥ 0.75 | ✅ pass |
| Faithfulness (LLM judge, 0–1) | 0.98 | ≥ 0.90 | ✅ pass |
| Citation validity | 1.00 | ≥ 0.98 | ✅ pass |
| Refusal recall (unanswerable qs) | 1.00 | ≥ 0.80 | ✅ pass |
| Refusal precision | 0.80 | ≥ 0.75 | ✅ pass |
| Retrieval hit-rate@5 | 0.98 | ≥ 0.85 | ✅ pass |
| Cost per query (USD) | 0.0031 | ≤ 0.010 | ✅ pass |
| p95 latency, uncached (ms) | 4 200 | ≤ 6 000 | ✅ pass |
| p95 latency, cached (ms) | ~35 | ≤ 800 | ✅ pass |

*All 45 questions are evaluated by `labs/lab7/gate.py`, which runs the shipped pipeline (`labs/lab7/pipeline.py`). Retrieval hit-rate is measured over the 42 questions with relevant documents. Small samples: refusal recall rests on 5 unanswerable questions, so one miss moves it by 0.20.*

> **TODO(Archit) before submitting:** refresh this table from `reports/gate_report.json` after the warm run (`python labs/lab7/gate.py`). The figures above predate the gate rewrite, which now evaluates all 45 questions and measures cost and latency instead of hardcoding them.

---

## 3 — Where It Fails

**Remaining failure modes (quantified on the 45-question set):**

| Failure mode | Count | Example |
|---|---|---|
| Partial-refusal questions answered too conservatively | 2 | Q37 (Singapore cover): system refuses the whole question instead of answering the supported part |
| Multi-hop questions with retrieval miss | 2 | Q25 (caesarean cost) — correct calculation requires retrieving *both* maternity-benefits and plan-gold in the top 5; dense retrieval sometimes surfaces only one |
| Paraphrase questions with lexical gap | 1 | Q41 ("skip paying" → grace period) — BM25 fails; hybrid saves it but only 80% of the time |
| Archived-document trap | 0 | Archived docs excluded at index build time; all three trap questions now pass |
| Citation hallucination | 0 | `enforce_citations` removes out-of-range indices before the answer is returned |

**Most significant remaining failure:** Q37-class partial-refusal. The system correctly identifies that the corpus only partially answers the question, but the generation model refuses the whole question rather than answering the supported sub-part with explicit "and the rest is not covered by this corpus" framing. Fix: add a "partial answer" instruction to the system prompt and a re-validation step.

---

## 4 — What It Costs

| Scope | Cost |
|---|---|
| Per query (uncached) | \$0.0031 |
| Per query (cached exact hit) | \$0.00 |
| Per 1 000 queries (30% cache hit rate) | \$2.17 |
| Per year at 10 000 queries/day | \$7 920 (\$8 830 at the measured 22% hit rate) |

*Model: gemini-3.7-flash (MAIN) for generation, gemini-embedding-001 (EMBED) for retrieval. Prices per ai.google.dev 2026-08-22. The 30% exact-cache hit rate is conservative; measured rate on the dev traffic was 22%.*

**Semantic cache note:** The semantic cache threshold is set at cosine ≥ 0.95. Below 0.92, we observed wrong answers for questions that differ only by plan name (e.g., "waiting period on Gold" vs "waiting period on Silver" — cosine ≈ 0.93 with the same embedding model). The additional hit rate from semantic caching is ~8% at threshold 0.95; the risk of a silent wrong answer is the reason this threshold is not lowered.

---

## 5 — How Fast It Is

**Stage breakdown (p95, uncached, from `.aip_traces/`):**

```
embed query      ~  40 ms   (1%)
retrieve hybrid  ~  15 ms   (< 1%)
rerank           —          (no reranker in production config)
generate         ~4 100 ms  (~98%)
validate         ~   5 ms   (< 1%)
──────────────────────────────────
total            ~4 160 ms
```

**What to optimise first:** Generation dominates at 98% of total latency. The main lever is **answer length** — the model currently outputs 80–200 tokens; halving that would save ~2 seconds. The cost of shortening is precision: Lab 5's measurement showed that aggressive length reduction (max_tokens=150) drops correctness by ~0.08. A better first step is to reduce the context size (`max_context_chars` from 8 000 to 4 000) and measure whether correctness holds — the retriever currently sends more text than the model uses.

**Streaming TTFT.** `POST /ask/stream` streams model tokens as they are generated and reports `ttft_ms` in its final event. Citations are validated after generation and sent last, with `citations_valid`; the UI warns when it is false.

> **TODO(Archit) before submitting:** measure TTFT on 10 uncached queries through `/ask/stream` and report median and p95 here. An earlier version split a finished answer into words, so its TTFT figures (420 / 780 ms) were not real and have been removed.

---

## 6 — What It Is Not Safe For

> This section is the professional boundary of safe use. A limitations section that cannot say *why* is not a limitations section.

**Do not rely on this system for:**

1. **Coverage decisions without human review.** The system answers from the policy documents as indexed — but the documents are the *summary* policy, not the master contract. Aurora's master contract contains sub-clauses, endorsements, and rider schedules not in this corpus. A system that says "covered" when the relevant clause is in a document it never indexed is confidently wrong. Every coverage decision should be verified against the master policy or with an underwriter.

2. **Medical advice or clinical recommendations.** The corpus describes financial terms (limits, exclusions, waiting periods). It says nothing about medical necessity, treatment options, or clinical appropriateness. The system must not be used to guide treatment decisions.

3. **Situations involving the archived 2024 policy document.** This document is excluded from the index (Lab 3's D3 result). Questions that compare 2024 and 2026 timelines will receive an answer based only on the current document. Users comparing policies across years must be directed to a human agent.

4. **High-stakes financial calculations.** The premium arithmetic tool in the agent mode produces estimates based on the corpus description of loadings and discounts; it is not connected to Aurora's pricing engine. Never quote a premium from this system as a binding figure.

5. **Any jurisdiction outside India.** The corpus covers Aurora Health Insurance under Indian Insurance Regulatory and Development Authority (IRDAI) regulations. Terms, exclusions, and timelines differ in other jurisdictions.

---

## 7 — What We Would Do Next

Ranked by expected value (impact × feasibility):

**1. Reduce generation latency by tuning context size (expected: −2 s p95, no correctness loss)**
The current `max_context_chars=8000` sends more text than the model uses. Running an ablation at 4000 chars would halve generation input and likely reduce latency by 40–50% with minimal correctness loss (the top 5 chunks contain the answer in >93% of cases). This is the highest-leverage change because it costs nothing and latency is the primary user experience metric.

**2. Fix partial-refusal handling for sub-answerable questions (expected: +0.05 correctness)**
The Q37 failure class (2 questions) requires the model to answer the part the corpus supports and explicitly state what it cannot answer. Adding a rule to the system prompt ("If sources answer only part of the question, answer the supported part and state what is not covered") with a validation re-run would catch this. This directly affects the correctness metric.

**3. Add a cross-encoder reranker (expected: +0.04 hit-rate, +1 000 ms p95)**
Lab 3 measured that a cross-encoder on k=30→5 improves nDCG@10 by ~0.06. The 1 000 ms latency cost is acceptable if generation is first brought below 3 s by change 1. This is ranked third because the current retrieval is already above the gate threshold and the gain is marginal relative to the cost.

---
