"""Shared core for the Company Policy Assistant.

Three approaches that all return the same PolicyResult:
  1. rules_based   - TF-IDF keyword search, returns the policy text verbatim (no LLM)
  2. llm_no_index  - Gemini gets the ENTIRE policy database in the prompt
  3. llm_vector    - policies are embedded once; the top-k most similar are sent to Gemini (RAG)
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "company_policies.csv"
EMBED_CACHE = BASE_DIR / "policy_embeddings.npy"

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
TOP_K = 3
RULES_MIN_SCORE = 0.15  # below this similarity the rules engine says "no matching policy"

NO_ANSWER = "I could not find a company policy that answers this question."


# ---------------------------------------------------------------- data

@dataclass
class Policy:
    title: str
    department: str
    policy_text: str
    category: str

    def as_text(self) -> str:
        return f"{self.title} ({self.category}, owner: {self.department}): {self.policy_text}"


@dataclass
class PolicyResult:
    approach: str
    answer: str
    policy_title: str | None      # policy the approach cites (None = no policy found)
    response_time_s: float
    input_tokens: int
    output_tokens: int
    policy_exists: bool           # does the cited title exist in the database?
    claims_supported: bool        # does the approach itself say the answer is backed by a policy?

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


def load_policies(path: Path = CSV_PATH) -> list[Policy]:
    with open(path, encoding="utf-8") as f:
        return [Policy(**{k: v.strip() for k, v in row.items()}) for row in csv.DictReader(f)]


POLICIES = load_policies()
TITLES = {p.title for p in POLICIES}


# ---------------------------------------------------------------- 1. rules-based

_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
_tfidf_matrix = _vectorizer.fit_transform(
    f"{p.title} {p.title} {p.category} {p.policy_text}" for p in POLICIES  # title weighted x2
)


def rules_based(question: str) -> PolicyResult:
    start = time.perf_counter()
    scores = cosine_similarity(_vectorizer.transform([question]), _tfidf_matrix)[0]
    best = int(np.argmax(scores))
    elapsed = time.perf_counter() - start

    if scores[best] < RULES_MIN_SCORE:
        return PolicyResult("Rules-based search", NO_ANSWER, None, elapsed, 0, 0, False, False)
    p = POLICIES[best]
    return PolicyResult("Rules-based search", p.policy_text, p.title, elapsed, 0, 0, True, True)


# ---------------------------------------------------------------- Gemini helpers

_client = None


def _gemini():
    global _client
    if _client is None:
        from google import genai
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in .env or Streamlit secrets).")
        from google.genai import types
        _client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60_000))
    return _client


SYSTEM_PROMPT = (
    "You are a company policy assistant. Answer ONLY using the policies provided. "
    "Cite exactly one policy by its exact title. If none of the provided policies answers "
    "the question, set policy_title to null, supported to false, and say you could not find "
    "a relevant policy. Keep the answer to 1-2 sentences."
)

_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answer": {"type": "STRING"},
        "policy_title": {"type": "STRING", "nullable": True},
        "supported": {"type": "BOOLEAN"},
    },
    "required": ["answer", "policy_title", "supported"],
}


def _with_retry(call, attempts: int = 6):
    """Retry on Gemini overload/rate-limit errors.

    Returns (result, seconds lost to failed attempts and back-off) so callers can
    report the latency of the successful call only.
    """
    import re
    import httpx
    from google.genai import errors
    waited = 0.0
    for i in range(attempts):
        t0 = time.perf_counter()
        try:
            return call(), waited
        except (errors.APIError, httpx.TransportError) as e:
            waited += time.perf_counter() - t0
            transient = isinstance(e, httpx.TransportError) or e.code in (429, 500, 503, 504)
            if not transient or i == attempts - 1:
                raise
            # A 429 (free-tier quota) tells us how long to wait: "Please retry in 48.3s"
            hint = re.search(r"retry in ([\d.]+)s", str(e))
            pause = min(float(hint.group(1)) + 1, 65) if hint else 2 ** i
            time.sleep(pause)
            waited += pause


def _ask_gemini(approach: str, question: str, policies: list[Policy], start: float,
                extra_input_tokens: int = 0, waited: float = 0.0) -> PolicyResult:
    from google.genai import types

    context = "\n".join(f"- {p.as_text()}" for p in policies)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=_RESPONSE_SCHEMA,
        temperature=0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    resp, w = _with_retry(lambda: _gemini().models.generate_content(
        model=CHAT_MODEL, contents=f"POLICIES:\n{context}\n\nQUESTION: {question}", config=config,
    ))
    elapsed = time.perf_counter() - start - waited - w  # retry back-off is not model latency

    data = json.loads(resp.text)
    title = data.get("policy_title") or None
    usage = resp.usage_metadata
    return PolicyResult(
        approach=approach,
        answer=data.get("answer", "").strip(),
        policy_title=title,
        response_time_s=elapsed,
        input_tokens=(usage.prompt_token_count or 0) + extra_input_tokens,
        # thinking tokens are billed as output, so count them too
        output_tokens=(usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
        policy_exists=title in TITLES,
        claims_supported=bool(data.get("supported")),
    )


# ---------------------------------------------------------------- 2. LLM without vector index

def llm_no_index(question: str) -> PolicyResult:
    start = time.perf_counter()
    return _ask_gemini("LLM without vector index", question, POLICIES, start)


# ---------------------------------------------------------------- 3. LLM with vector index

_policy_vectors: np.ndarray | None = None


def _embed(texts: list[str], task: str) -> tuple[np.ndarray, float]:
    from google.genai import types
    resp, waited = _with_retry(lambda: _gemini().models.embed_content(
        model=EMBED_MODEL, contents=texts, config=types.EmbedContentConfig(task_type=task)
    ))
    vecs = np.array([e.values for e in resp.embeddings], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True), waited


def build_index(force: bool = False) -> np.ndarray:
    """Embed every policy once and cache the vectors on disk (policy_embeddings.npy)."""
    global _policy_vectors
    if _policy_vectors is not None and not force:
        return _policy_vectors
    if EMBED_CACHE.exists() and not force:
        cached = np.load(EMBED_CACHE)
        if len(cached) == len(POLICIES):
            _policy_vectors = cached
            return cached
    vecs = np.vstack([
        _embed([p.as_text() for p in POLICIES[i:i + 50]], "RETRIEVAL_DOCUMENT")[0]
        for i in range(0, len(POLICIES), 50)  # API accepts max 100 per batch
    ])
    np.save(EMBED_CACHE, vecs)
    _policy_vectors = vecs
    return vecs


def retrieve(question: str, k: int = TOP_K) -> tuple[list[Policy], float]:
    vecs = build_index()
    q, waited = _embed([question], "RETRIEVAL_QUERY")
    top = np.argsort(vecs @ q[0])[::-1][:k]
    return [POLICIES[i] for i in top], waited


def llm_vector(question: str) -> PolicyResult:
    build_index()  # one-off indexing cost is not part of per-question latency
    # Embedding the question also costs tokens; count them outside the timer.
    try:
        counted, _ = _with_retry(lambda: _gemini().models.count_tokens(model=EMBED_MODEL, contents=question))
        embed_tokens = counted.total_tokens or 0
    except Exception:
        embed_tokens = max(1, len(question) // 4)  # rough estimate: ~4 characters per token
    start = time.perf_counter()
    hits, waited = retrieve(question)
    return _ask_gemini("LLM with vector index", question, hits, start,
                       extra_input_tokens=embed_tokens, waited=waited)


# ---------------------------------------------------------------- convenience

APPROACHES = {"rules": rules_based, "no_index": llm_no_index, "vector": llm_vector}


def ask_all(question: str) -> list[PolicyResult]:
    return [fn(question) for fn in APPROACHES.values()]


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How many days per week can I work from home?"
    print(f"Q: {q}\n")
    for name, fn in APPROACHES.items():
        try:
            r = fn(q)
            print(f"[{r.approach}] {r.policy_title} | {r.response_time_s:.2f}s | "
                  f"{r.total_tokens} tok\n  {r.answer}\n")
        except Exception as e:
            print(f"[{name}] ERROR: {e}\n")
