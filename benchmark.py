"""Run every test question through all three approaches and score them.

Question types (test_questions.csv):
  direct       - answer is in one policy, question shares keywords with it
  paraphrase   - answer is in one policy, but worded differently
  gap          - a related policy exists but does NOT contain the detail asked for
  not_covered  - no policy addresses the question; the right behaviour is to abstain

Scores per answer:
  policy_correct - cited policy == expected policy (for not_covered: cited no policy)
  unsupported    - LLM judge: the answer states something the policy database does not
                   support (invented facts, or an unrelated policy presented as the answer)

Output: benchmark_results.csv (one row per question x approach) and benchmark_summary.csv.
Usage:  python benchmark.py            run everything (resumes after a crash)
        python benchmark.py --rescore  recompute the summary from benchmark_results.csv
"""
import csv
import json
import os
import sys
import time

import pandas as pd
from google.genai import types

import policy_core as core

# A different model judges, so the answering model does not grade its own work
# (and on the free tier it has its own 15 requests/minute quota).
JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.1-flash-lite")
SECONDS_PER_QUESTION = 14  # free tier: 15 req/min per model; the judge makes 3 calls per question

JUDGE_PROMPT = """You are auditing a company policy assistant.

POLICY DATABASE (the only source of truth):
{db}

QUESTION: {question}
ASSISTANT'S CITED POLICY: {policy}
ASSISTANT'S ANSWER: {answer}

Is the assistant's answer UNSUPPORTED by the policy database? Mark it unsupported if EITHER:
 (a) it states any fact, number, rule or permission that is not in the database, or
 (b) it presents a policy as answering the question when that policy does not actually address it.
Saying that no policy covers the question, or that a policy does not specify a detail, is SUPPORTED.
Return JSON: {{"unsupported": true/false, "reason": "<one short sentence>"}}"""

DB_TEXT = "\n".join(f"- {p.as_text()}" for p in core.POLICIES)


def judge(question: str, r: core.PolicyResult) -> tuple[bool, str]:
    prompt = JUDGE_PROMPT.format(db=DB_TEXT, question=question,
                                 policy=r.policy_title or "none", answer=r.answer)
    resp, _ = core._with_retry(lambda: core._gemini().models.generate_content(
        model=JUDGE_MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0,
            response_schema={"type": "OBJECT", "required": ["unsupported", "reason"], "properties": {
                "unsupported": {"type": "BOOLEAN"}, "reason": {"type": "STRING"}}},
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)),
    ))
    data = json.loads(resp.text)
    return bool(data.get("unsupported")), data.get("reason", "")


APPROACH_ORDER = ["Rules-based search", "LLM without vector index", "LLM with vector index"]


def load_questions() -> list[dict]:
    with open(core.BASE_DIR / "test_questions.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    # Labels always come from test_questions.csv, so relabelling only needs --rescore.
    labels = pd.DataFrame(load_questions()).replace({"expected_policy": {"": None}})
    df = df.drop(columns=["type", "expected_policy", "policy_correct"], errors="ignore").merge(labels, on="question")
    df["policy_correct"] = [(t if isinstance(t, str) else None) == (e if isinstance(e, str) else None)
                            for t, e in zip(df["policy_title"], df["expected_policy"])]
    df.to_csv(core.BASE_DIR / "benchmark_results.csv", index=False)

    by = lambda d, col: d.groupby("approach")[col].mean()
    summary = pd.DataFrame({
        "Correct policy: direct (8)": by(df[df.type == "direct"], "policy_correct"),
        "Correct policy: paraphrased (8)": by(df[df.type == "paraphrase"], "policy_correct"),
        "Correct policy: detail missing (3)": by(df[df.type == "gap"], "policy_correct"),
        "Correct abstention: no policy (5)": by(df[df.type == "not_covered"], "policy_correct"),
        "Unsupported answers (all 24)": by(df, "unsupported"),
        "Median response time (s)": df.groupby("approach")["response_time_s"].median(),
        "Avg tokens per question": by(df, "total_tokens"),
    }).reindex(APPROACH_ORDER)
    summary.to_csv(core.BASE_DIR / "benchmark_summary.csv")
    return summary


def run():
    core.build_index()
    questions = load_questions()

    # Resume support: every answered row is appended to a partial file immediately.
    partial = core.BASE_DIR / "benchmark_results.partial.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    done = {(r["question"], r["approach"]) for r in rows}

    for i, q in enumerate(questions, 1):
        q_start = time.monotonic()
        for name, fn in zip(APPROACH_ORDER, core.APPROACHES.values()):
            if (q["question"], name) in done:
                continue
            r = fn(q["question"])
            unsupported, reason = judge(q["question"], r)
            rows.append({"question": q["question"], **r.to_dict(),
                         "unsupported": unsupported, "judge_reason": reason})
            pd.DataFrame(rows).to_csv(partial, index=False)
            ok = r.policy_title == (q["expected_policy"] or None)
            print(f"[{i}/{len(questions)}] {r.approach:26} {'OK ' if ok else 'BAD'} "
                  f"{'UNSUPPORTED' if unsupported else '':11} {r.response_time_s:5.2f}s "
                  f"{r.total_tokens:5} tok  {q['question'][:50]}", flush=True)
        time.sleep(max(0.0, SECONDS_PER_QUESTION - (time.monotonic() - q_start)))

    partial.unlink()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(core.BASE_DIR / "benchmark_results.csv") if "--rescore" in sys.argv else run()
    print(summarize(df).round(3).to_string())
