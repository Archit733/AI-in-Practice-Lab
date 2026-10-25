#!/usr/bin/env python3
"""B1: find the semantic-cache threshold where wrong answers start.

    python labs/lab7/sem_sweep.py              # embeds the pairs (pennies), caches them
    AIP_OFFLINE=1 python labs/lab7/sem_sweep.py   # replays once cached

Two kinds of question pairs:
  SAME      paraphrases -- should share one cached answer (a hit is good)
  DIFFERENT near-misses that need different answers -- a hit is a WRONG answer

For each threshold t, a pair "hits" if cosine >= t. We report:
  paraphrase hit rate  (higher = more cache savings)
  wrong-hit rate       (must be 0 -- any hit here serves the wrong answer)
The safe threshold is the lowest t with zero wrong hits; report it, and the
most similar DIFFERENT pair, which is what goes wrong just below it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SAME = [
    ("How long do I have to file a reimbursement claim?",
     "What is the deadline for submitting a reimbursement claim?"),
    ("Is maternity covered on the Bronze plan?",
     "Does the Bronze plan include maternity cover?"),
    ("What is the room rent limit on Silver?",
     "How much room rent does the Silver plan allow?"),
    ("What happens if I miss a premium payment?",
     "What if I don't pay my premium on time?"),
    ("Which hospitals are in the cashless network?",
     "Where can I get cashless treatment?"),
    ("How do I add my spouse to my policy?",
     "Can I include my wife or husband in my cover?"),
]

DIFFERENT = [
    ("What is the waiting period on Gold?", "What is the waiting period on Silver?"),
    ("What is the room rent limit on Silver?", "What is the room rent limit on Gold?"),
    ("Is maternity covered on Bronze?", "Is maternity covered on Platinum?"),
    ("What is the co-payment for senior citizens?", "What is the co-payment for children?"),
    ("How long is the grace period for monthly premiums?",
     "How long is the grace period for annual premiums?"),
    ("What is the claim deadline for cashless treatment?",
     "What is the claim deadline for reimbursement?"),
    ("Is knee replacement covered in the first year?",
     "Is knee replacement covered after two years?"),
]

THRESHOLDS = [0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]


def main() -> int:
    from aip.embed import embed_batch

    def cos(pairs):
        a = embed_batch([p[0] for p in pairs], input_type="query")
        b = embed_batch([p[1] for p in pairs], input_type="query")
        return [float(x @ y) for x, y in zip(a, b)]

    same, diff = cos(SAME), cos(DIFFERENT)

    print("Most similar pairs that need DIFFERENT answers:")
    for s, (q1, q2) in sorted(zip(diff, DIFFERENT), reverse=True)[:3]:
        print(f"  {s:.3f}  {q1!r}  vs  {q2!r}")
    print("Least similar paraphrases (should still hit):")
    for s, (q1, q2) in sorted(zip(same, SAME))[:3]:
        print(f"  {s:.3f}  {q1!r}  vs  {q2!r}")

    print(f"\n{'threshold':>9}  {'paraphrase hits':>15}  {'WRONG hits':>10}")
    safe = None
    for t in THRESHOLDS:
        ph = sum(s >= t for s in same) / len(same)
        wh = sum(s >= t for s in diff)
        print(f"{t:>9.2f}  {ph:>15.0%}  {wh:>10}")
        if wh == 0 and safe is None:
            safe = t
    print(f"\nLowest threshold with zero wrong hits: {safe}")
    print("Pick at or above it, and keep monitoring refusals and thumbs-down reports.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
