# Lab 7 — Technical Implementation & Engineering Report (Parts A to E)

**Course:** AI in Practice I — Module 1: Applied GenAI  
**Deliverable:** Comprehensive Engineering Walkthrough covering Parts A through E  
**Repository:** `aip-lab1` | **Service:** Aurora Policy Assistant

---

## Executive Summary

Lab 7 ("Ship It") transforms individual RAG and agent components from Labs 1–6 into a production-ready, audited HTTP service. This report details the technical architecture, design decisions, failure analyses, empirical thresholds, and trade-offs made across each stage:
- **Part A:** The HTTP Service Architecture & Contracts (`service.py`, `ui.py`)
- **Part B:** Caching Hierarchy, Streaming Protocol, and the Latency Budget
- **Part C:** Observability, Tracing Topology, and Silent Failure Alerting (`dashboard.py`)
- **Part D:** Continuous Integration Regression Gating (`gate.py`, `thresholds.yml`)
- **Part E:** Evaluation Synthesis and Operational Boundary Defense

---

## Part A — The Service Architecture

### A1. API Contract Design & Philosophy (`POST /ask`)
The production endpoint `POST /ask` is exposed via FastAPI with strict Pydantic v2 schemas.

```python
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
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
    trace_id: str
```

#### Why `cost_usd` and `trace_id` are in the Response Body
In standard web engineering, infrastructure metadata is often relegated to headers or internal logging. Here, returning `cost_usd` and `trace_id` in the JSON response body serves two critical operational goals:
1. **Production Debuggability:** When an end-user or support agent reports an erroneous or hallucinated answer from days prior, the `trace_id` gives the debugging engineer a direct $O(1)$ key into `.aip_traces/*.jsonl` rather than requiring timestamp grepping and guessing.
2. **Economic Transparency:** The user or tenant footing the API bill directly inspects the cost of each query in real time, preventing silent budget leakage.

---

### A2. Pipeline Wiring & Startup Caching
A common failure mode in naive RAG deployments is instantiating the index and re-embedding the corpus on every incoming HTTP request (causing 30–40 s p95 spikes).

In `labs/lab7/service.py`:
- The pipeline is constructed **once at application startup** (`pipeline()`) and cached in the global variable `_PIPELINE`.
- **Retrieval Engine:** Markdown chunking at 400 characters (the winning configuration from Lab 3), with archived documents explicitly excluded to eliminate historical policy confusion (Q29–Q31).
- **Hybrid Fusion:** Reciprocal Rank Fusion (`HybridRetriever`) combining exact cosine search (`DenseRetriever`) with lexical search (`Bm25Retriever`) at $k=60$.
- **Guard Layer Integration:** Untrusted context delimiters (`delimit_untrusted`) and citation range validation (`enforce_citations`) are hard-wired directly into the generation and validation flow.

---

### A3. Status Code Semantics & Graceful Degradation
The service enforces semantic status codes that communicate actionable next steps to the client:

| HTTP Status | Trigger | Underlying Cause | Client Recovery Action |
|---|---|---|---|
| **422 Unprocessable Entity** | Request validation failure | Question too short ($<3$ chars) or missing fields | Fix client payload; do not retry unchanged. |
| **429 Too Many Requests** | `BudgetExceeded` | Spent ceiling exceeded ($AIP\_BUDGET\_USD$) | Back off; request quota increase. |
| **503 Service Unavailable** | Upstream provider outage / rate limit | LLM upstream network timeout, 502/503 from provider | Retry automatically after the duration in `Retry-After: 30`. |
| **500 Internal Server Error** | Genuine unhandled software defect | Code crash | Bug report; do not retry. |

> **Key Rule:** Provider outages must **never** leak as bare 500s with stack traces. Returning 500 causes naive client libraries to retry immediately and aggressively, compounding provider rate limits into retry storms.

---

### A4. Front-End Interface (`labs/lab7/ui.py`)
Grounding is only meaningful if the reader can inspect the backing evidence. The Streamlit UI provides:
- Rendered markdown answers with warning callouts for legitimate policy refusals.
- **Expandable Source Blocks:** Each citation $[i]$ expands to display the exact raw excerpt extracted from the corpus file.
- Real-time diagnostic bar displaying latency, exact dollar cost, cache status, and clickable trace IDs.
- **Review Queue Button (Stretch):** Captures problematic interactions into `.aip_review_queue/` for continuous golden set curation.

---

## Part B — Caching, Streaming & Latency Budget

### B1. Two-Tier Caching Hierarchy

```
Query -> [Exact Response Cache (SHA-256)] -> HIT (0 ms, $0.00)
                     |
                   MISS
                     v
         [Semantic Cache (Embedding Cosine >= 0.95)] -> HIT (40 ms, $0.00)
                     |
                   MISS
                     v
         [Execute RAG Pipeline (Retrieve -> Generate -> Validate)]
```

#### Exact Cache vs. Semantic Cache Comparison

| Dimension | Exact Response Cache | Semantic Cache |
|---|---|---|
| **Key Generation** | SHA-256 hash of normalized text (`" ".join(q.lower().split())`) | Dense embedding vector of question |
| **Storage Engine** | Content-addressed SQLite (`calls.sqlite3`) | In-memory cosine similarity matrix |
| **Hit Rate** | 15% – 30% | Additional +10% – 15% |
| **Failure Risk** | **Zero.** Same normalized input guarantees identical intent. | **High.** Can return answers to completely different questions. |

#### Empirical Semantic Cache Threshold Determination
A semantic cache can silently compromise RAG reliability because below the safe threshold, it returns a confident, fast, well-cited answer to a question the user never asked.

**Measured Failure Case:**
```text
Question A: "What is the waiting period on Gold?"
Question B: "What is the waiting period on Silver?"
Cosine Similarity: ~0.932
```
- At threshold **0.90 – 0.92**: Question B hits the cached answer for Question A. Latency drops to ~40 ms and cost is $0.00, but the customer is told the wrong waiting period.
- At threshold **$\ge 0.95$**: Distinguishes plan-specific and nuance variations while successfully clustering true paraphrases (e.g., *"How long before I can claim?"* vs *"What is the claim filing window?"*).
- **Production Choice:** Threshold pinned to **0.95**.

---

### B2 & B3. Streaming & The Validation Dilemma

#### The Problem
RAG models generate output tokens sequentially. A 4-second completion feels like an unresponsive system unless streaming is used. However, **you cannot validate citation ranges until the answer is fully generated, but tokens have already been delivered to the client.**

#### Strategy Evaluation

| Strategy | Architecture | Cost / Trade-off | Evaluation |
|---|---|---|---|
| **1. Buffer, Validate, then Stream** | Run full generation, execute `enforce_citations`, then stream the buffered tokens to the UI. | Destroys the Time-To-First-Token (TTFT) benefit entirely. TTFT equals total generation time. | Rejected. |
| **2. Stream with Post-Validation Event** | Stream immediately; append an error or retraction event if citations fail validation. | High UI complexity. The user has already read the invalid claim before the retraction arrives. | Rejected for high-stakes domains. |
| **3. Stream Prose, Hold Citations to End** | Stream answer tokens immediately (low TTFT); hold structured citations and metadata until generation finishes; deliver as the terminal SSE event. | Grounding sources appear ~1–2 seconds after reading begins. | **Selected & Defended.** |

**Defense of Strategy 3:**  
In user research and practice, human reading speed is ~200–250 words per minute. Delivering prose immediately satisfies the user's perception of speed: tokens stream from the model as they are generated, and TTFT is reported in the final event (re-measure on your key; the earlier 780 ms figure came from a simulated stream). By the time the user finishes reading the sentence, the terminal SSE event delivers the verified citations, rendering the expandable source boxes without race conditions.

---

### B4. Latency Budget Decomposition

Measurement over uncached queries on the production profile (`gemini-3.7-flash` generation, `gemini-embedding-001` retrieval):

```
Stage                   Latency (p95)    Fraction
─────────────────────────────────────────────────
1. Query Embedding           ~40 ms        ~1.0%
2. Hybrid Retrieval          ~15 ms        ~0.4%
3. Reranking                  —            —
4. LLM Generation         ~4,100 ms       ~98.4%
5. Citation Validation        ~5 ms        ~0.1%
─────────────────────────────────────────────────
Total End-to-End          ~4,160 ms      100.0%
```

#### What to Optimize First
Generation consumes **98.4%** of end-to-end latency. Retrieval and embedding optimizations are irrelevant until generation is addressed.
- **Top optimization:** Restrict retrieved context characters (`max_context_chars`) from 8,000 to 4,000. Ingesting fewer prompt tokens reduces internal model attention passes and cuts completion latency by an estimated 30–40% without harming accuracy.

---

## Part C — Observability & Alerting

### C1. Distributed Tracing Topology (`aip.tracing`)
Every invocation instruments nested spans written to `.aip_traces/<run_id>.jsonl`:

```
http.ask [span_id: 8f3a1, duration: 4120ms]
  ├── rag.answer [parent: 8f3a1]
  │     ├── rag.retrieve [duration: 15ms]
  │     │     ├── retrieve.dense [duration: 12ms]
  │     │     └── retrieve.bm25 [duration: 2ms]
  │     └── rag.generate [duration: 4095ms]
  │           └── llm.call [tier: MAIN, tokens: 412in/115out]
  └── guard.enforce_citations [duration: 1ms]
```

This structure immediately answers: *"Why did request X take 9 seconds?"*  
The trace will explicitly pinpoint whether it was a reranking backoff, an upstream token throttle, or a prompt injection retry loop.

---

### C2 & C3. Operations Dashboard (`labs/lab7/dashboard.py`)
Built with Streamlit over local trace logs:
- **KPI Metrics:** Total spans, aggregate spend, LLM call counts, cache hit percentages, and active error counts.
- **Stage Breakdown Table:** Aggregates p50, p95, p99, and total duration grouped by span name.
- **Cost Accumulation:** Running cumulative expenditure chart.
- **Rolling Latency:** 5-minute rolling p95 window.

---

### C4. Alert Design: Refusal Rate Doubling
An alert must detect failures before customers do, with actionable remediation instructions.

- **Alert Condition:** Refusal rate doubles relative to a 30-minute rolling baseline.
- **Why This Alert Matters:**  
  When a database or vector index breaks (e.g., deleted collection, corrupted embeddings), typical infrastructure alerts stay silent:
  - Error rate does not rise ($0$ exceptions thrown).
  - Latency does not spike (empty searches are faster).
  - Cost does not increase.
  
  However, a well-engineered RAG pipeline responds to empty context by correctly declining: *"I don't have enough information in the provided sources to answer that."* Therefore, **a doubling in refusal rate is the primary signature of a silent index failure.**

- **Runbook / Action Plan When Fired:**
  1. Inspect `rag.retrieve` spans in trace files; check `n_hits`.
  2. If `n_hits == 0`, re-index corpus documents via `scripts/warm_cache.py`.
  3. If `n_hits > 0`, check for document drift or recent updates to `data/corpus/`.

---

## Part D — The Regression Gate

### D1 & D2. CI Gate Design (`gate.py`, `thresholds.yml`)
The regression gate prevents degraded pipelines from reaching production. It runs the golden set (`data/eval/rag_golden.jsonl`) against quality, safety, and economic ceilings:

```yaml
correctness:        {min: 0.75}
faithfulness:       {min: 0.90}
citation_validity:  {min: 0.98}
refusal_recall:     {min: 0.80}
refusal_precision:  {min: 0.75}
hit_rate_at_5:      {min: 0.85}
cost_per_query_usd: {max: 0.010}
p95_latency_ms:     {max: 6000}
```

#### Offline Determinism (`AIP_OFFLINE=1`)
In `.github/workflows/eval.yml`, CI runs with `AIP_OFFLINE=1` against the committed `.aip_cache/calls.sqlite3`.
- **Cost in CI:** $0.00.
- **Network dependency:** Zero.
- **Determinism:** the gate replays the shipped pipeline through the cache, so CI failures reflect code or config changes, not provider drift. A changed prompt cannot hit the cache, which is what makes D3 detectable offline.
- **Workflow location:** GitHub only runs workflows from `.github/workflows/` at the repository root; the live file is there, not under `aip-lab1/`.

---

### D3. The Deliberate Failure Verification
> *"A gate you have not seen fail is a gate you do not have."*

To prove the gate is active and functional:
1. Break the *system*, not the gate: set `FINAL_K = 1` in `labs/lab7/pipeline.py` (the single definition the service and the gate share).
2. Execute gate, exactly as CI does:
   ```bash
   AIP_OFFLINE=1 python3 labs/lab7/gate.py
   ```
3. What happens: the generator now sees one chunk instead of five, so every generation prompt changes and none of them is in the committed cache (`cache_coverage` 0.0, FAIL). Retrieval is still measured locally, and the share of questions whose context contains a relevant document falls sharply (`hit_rate_at_5`, FAIL). Exit code 1.
4. This confirms non-zero exit codes propagate to CI runners, blocking broken pull requests.

---

## Part E — Evaluation & Deployment Synthesis

### 1. Verified Golden Set Performance

```text
metric                   value            gate  status
----------------------------------------------------------
correctness             0.9000         >= 0.75  ok
faithfulness            0.9778          >= 0.9  ok
citation_validity       1.0000         >= 0.98  ok
refusal_recall          1.0000          >= 0.8  ok
refusal_precision       0.8000         >= 0.75  ok
hit_rate_at_5           0.9762         >= 0.85  ok
cost_per_query_usd      0.0031         <= 0.01  ok
p95_latency_ms       4200.0000         <= 6000  ok
```

---

### 2. Failure Mode Breakdown

1. **Partial Refusal Over-Conservatism (Count: 2):**
   - *Example (Q37):* "Does Aurora cover treatment in Singapore, and up to what limit?"
   - The corpus notes emergency international cover exists on Platinum, but refers details to an addendum not present in the index. The system refuses the entire question rather than confirming what is covered and declining the rest.
2. **Multi-Hop Retrieval Misses (Count: 2):**
   - *Example (Q25):* Caesarean out-of-pocket costs on Gold. Requires both `maternity-benefits` and `plan-gold`. Dense retrieval occasionally drops one out of the top-5 window.
3. **Lexical Paraphrase Disconnect (Count: 1):**
   - *Example (Q41):* User asks about *"skipping paying on time"*. Lexical matching misses "grace period"; hybrid retrieval recovers it in 80% of runs.

---

### 3. Economic Profile

- **Unit Query Cost:** $\$0.0031$ (uncached) | $\$0.00$ (cache hit)
- **At 30% Cache Hit Rate:** $\$2.17$ per 1,000 queries
- **Enterprise Scale (10,000 queries/day):** $\$7,920$ per year ($\$8,830$ at the measured 22% hit rate)

---

### 4. Operational Boundaries & Safe Use
The system is explicitly **not safe** for:
1. **Binding Coverage Determinations:** Master policy documents contain legal endorsements not in summary markdown files.
2. **Clinical Decision Making:** No medical diagnostics or clinical validity checks exist.
3. **Multi-Year Comparisons:** The index excludes 2024 archived guidelines; comparisons across policy years require human customer service agents.

---

## Deliverables Index

| Deliverable | Path |
|---|---|
| Service Implementation | [`labs/lab7/service.py`](file:///Users/architbhatia/Downloads/AI-in-Practice-Lab/aip-lab1/labs/lab7/service.py) |
| Streamlit Front End | [`labs/lab7/ui.py`](file:///Users/architbhatia/Downloads/AI-in-Practice-Lab/aip-lab1/labs/lab7/ui.py) |
| Observability Dashboard | [`labs/lab7/dashboard.py`](file:///Users/architbhatia/Downloads/AI-in-Practice-Lab/aip-lab1/labs/lab7/dashboard.py) |
| Regression Gate Runner | [`labs/lab7/gate.py`](file:///Users/architbhatia/Downloads/AI-in-Practice-Lab/aip-lab1/labs/lab7/gate.py) |
| Threshold Specifications | [`labs/lab7/thresholds.yml`](file:///Users/architbhatia/Downloads/AI-in-Practice-Lab/aip-lab1/labs/lab7/thresholds.yml) |
| GitHub Actions Workflow | [`.github/workflows/eval.yml`](file:///Users/architbhatia/Downloads/AI-in-Practice-Lab/aip-lab1/.github/workflows/eval.yml) |
| Two-Page Evaluation Report | [`EVALUATION_REPORT.md`](file:///Users/architbhatia/Downloads/AI-in-Practice-Lab/aip-lab1/EVALUATION_REPORT.md) |
| Quickstart Run Guide | [`README.md`](file:///Users/architbhatia/Downloads/AI-in-Practice-Lab/aip-lab1/README.md) |
