"""A stand-in for `litellm`, for tests and keyless demos.

Put this directory first on PYTHONPATH and `from litellm import completion`
inside aip.llm / aip.embed resolves here instead of calling a real provider:

    PYTHONPATH=tests/fake_provider uvicorn labs.lab7.service:app

Everything above the provider -- the cache, the cost meter, retries,
tracing, the service, the gate -- runs for real. Only the model's replies are
synthetic, and they are deliberately simple and deterministic:

  * embeddings  hashed bag-of-words vectors (so retrieval is lexical-ish)
  * RAG answer  quotes the best-overlapping source and cites it, or refuses
                when no source shares enough content words with the question
  * judges      lexical-overlap heuristics, NOT a real quality judgement

Numbers produced with this provider demonstrate that the machinery works.
They say nothing about answer quality. Never quote them as results.

Knobs (environment variables):
  FAKE_LLM_OUTAGE=1        every call raises a 503-style provider error
  FAKE_LLM_TOKEN_DELAY_S   per-token delay (default 0.03)
  FAKE_LLM_BAD_CITATION=1  answers cite a source number that does not exist
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from types import SimpleNamespace

DIM = 256
_STOP = set("""a an the of to in on for and or is are was be do does did i my me we our you
your it its this that with at by from as can what which who when how much many any if
not no yes will would should could have has had get there their they them than then
about into after before under over per up down out so such only also just
aurora policy plan plans cover covered coverage insurance""".split())  # noqa: SIM905


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2]


def _vec(text: str) -> list[float]:
    v = [0.0] * DIM
    for w in _words(text):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 8) % 2 else -1.0
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def _maybe_outage() -> None:
    if os.getenv("FAKE_LLM_OUTAGE") == "1":
        raise RuntimeError("503 ServiceUnavailable: model overloaded (fake provider)")


# ---------------------------------------------------------------- embeddings
def embedding(model: str, input: list[str], **_: object):  # noqa: A002
    _maybe_outage()
    data = [{"embedding": _vec(t)} for t in input]
    usage = SimpleNamespace(prompt_tokens=sum(len(t.split()) for t in input))
    return SimpleNamespace(data=data, usage=usage)


# ---------------------------------------------------------------- chat
REFUSAL = "I don't have enough information in the provided sources to answer that."


def _rag_answer(user: str) -> str:
    q = user.split("Question:")[-1].split("Answer with citations:")[0]
    ctx = user.rsplit("Question:", 1)[0]
    ctx = re.sub(r"</?RETRIEVED_DOCUMENT[^>]*>", "", ctx)
    sources = re.findall(r"\[(\d+)\] \(source: [^)]+\)\n(.*?)(?=\n\[\d+\] \(source:|\Z)",
                         ctx, flags=re.S)
    qw = set(_words(q))
    best, best_score = None, 0.0
    for idx, text in sources:
        for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
            sent = sent.strip("#*|- ").strip()
            if len(sent) < 25:
                continue
            score = len(qw & set(_words(sent))) / max(1, len(qw))
            if score > best_score:
                best, best_score = (idx, sent), score
    if best is None or best_score < 0.5:
        return REFUSAL
    idx, sentence = best
    if os.getenv("FAKE_LLM_BAD_CITATION") == "1":
        idx = "9"
    sentence = sentence[:220].rstrip(".")
    return f"According to the policy documents, {sentence} [{idx}]."


def _judge(prompt: str) -> str:
    if "Compare a CANDIDATE answer to a REFERENCE answer" in prompt:
        ref = prompt.split("REFERENCE:")[1].split("CANDIDATE:")[0]
        cand = prompt.split("CANDIDATE:")[1].split("Reply as JSON")[0]
        ref_refuses = bool(re.search(r"not (in|covered|available|specified|stated)|"
                                     r"does not (say|state|specify|cover)|refuse|no information",
                                     ref, re.I))
        cand_refuses = cand.strip().startswith("I don't have enough information")
        if cand_refuses or ref_refuses:
            score = 2 if cand_refuses == ref_refuses else 0
        else:
            rw, cw = set(_words(ref)), set(_words(cand))
            j = len(rw & cw) / max(1, len(rw))
            score = 2 if j >= 0.5 else 1 if j >= 0.2 else 0
        return json.dumps({"score": score, "reason": "fake lexical judge"})
    if "grading whether an ANSWER is fully supported" in prompt:
        ctx = prompt.split("CONTEXT:")[1].split("ANSWER:")[0]
        ans = prompt.split("ANSWER:")[1].split("Reply as JSON")[0]
        aw = set(_words(ans)) - {"according", "policy", "documents"}
        support = len(aw & set(_words(ctx))) / max(1, len(aw))
        return json.dumps({"score": int(support >= 0.8), "unsupported_claims": [],
                           "reason": "fake lexical judge"})
    return json.dumps({"score": 1, "reason": "fake default"})


def _reply(messages: list[dict]) -> str:
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = messages[-1]["content"]
    if "answer questions using ONLY the numbered sources" in system:
        return _rag_answer(user)
    return _judge(user)


def _usage(messages, text):
    pt = int(sum(len(str(m.get("content", "")).split()) for m in messages) * 1.3)
    ct = int(len(text.split()) * 1.3) + 220   # + invisible "thinking" tokens, as on 3.7-flash
    return SimpleNamespace(prompt_tokens=pt, completion_tokens=ct)


def completion(model: str, messages: list[dict], stream: bool = False, **_: object):
    _maybe_outage()
    text = _reply(messages)
    delay = float(os.getenv("FAKE_LLM_TOKEN_DELAY_S", "0.03"))
    tokens = re.findall(r"\S+\s*", text)
    if not stream:
        time.sleep(delay * len(tokens) + 0.4)   # "thinking" before the answer
        msg = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=_usage(messages, text))

    def gen():
        time.sleep(0.4)
        for tok in tokens:
            time.sleep(delay)
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=tok))],
                                  usage=None)
        yield SimpleNamespace(choices=[], usage=_usage(messages, text))

    return gen()
