"""The shipped pipeline, defined once.

Both the service (service.py) and the regression gate (gate.py) build the
pipeline from here. That is the point: the gate must test the configuration
you actually ship. If they each build their own, the gate can pass while the
service is broken -- which is exactly what happened before this file existed.

D3 (prove the gate works): change FINAL_K to 1, or add a corpus doc to
EXCLUDED_DOCS, then run the gate. It must go red.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS_DIR = ROOT / "data/corpus"

# --- the shipped configuration ----------------------------------------------
CHUNK_SIZE = 400            # Lab 3 winner: markdown-aware chunks
RRF_K = 60                  # Reciprocal Rank Fusion constant
RETRIEVE_K = 12             # candidates pulled from hybrid search
FINAL_K = 5                 # chunks actually given to the generator
MAX_CONTEXT_CHARS = 8000    # B4: first thing to try lowering (to 4000)
TIER = "MAIN"
EXCLUDED_DOCS: set[str] = set()   # doc ids to drop (besides ARCHIVED ones)


def load_chunks():
    from aip.chunking import markdown_chunks

    corpus = {p.stem: p.read_text(encoding="utf-8")
              for p in sorted(CORPUS_DIR.glob("*.md"))}
    return [c for doc_id, text in corpus.items()
            if "ARCHIVED" not in doc_id and doc_id not in EXCLUDED_DOCS
            for c in markdown_chunks(text, doc_id, size=CHUNK_SIZE)]


def build_pipeline():
    """Hybrid (dense + BM25, RRF) retrieval + cited generation + Lab 6 guards.

    Guards wired in by aip.rag.RagPipeline:
      - delimit_untrusted: retrieved text is fenced as data, not instructions
      - enforce_citations: every [n] must index a real source
    """
    from aip.rag import RagPipeline
    from aip.retrieval import Bm25Retriever, DenseRetriever, HybridRetriever

    chunks = load_chunks()
    dense = DenseRetriever(chunks, show_progress=False)
    bm25 = Bm25Retriever(chunks)
    retriever = HybridRetriever([dense, bm25], rrf_k=RRF_K)
    return RagPipeline(
        retriever,
        k=RETRIEVE_K,
        final_k=FINAL_K,
        tier=TIER,
        max_context_chars=MAX_CONTEXT_CHARS,
    )


def config_dict() -> dict:
    return {
        "chunk_size": CHUNK_SIZE, "rrf_k": RRF_K, "retrieve_k": RETRIEVE_K,
        "final_k": FINAL_K, "max_context_chars": MAX_CONTEXT_CHARS,
        "tier": TIER, "excluded_docs": sorted(EXCLUDED_DOCS),
    }
