"""Streamlit website: compare the three policy-assistant approaches.

Run locally:  streamlit run app.py
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# On Streamlit Cloud the key lives in st.secrets; locally it comes from .env.
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ.setdefault("GEMINI_API_KEY", st.secrets["GEMINI_API_KEY"])
except FileNotFoundError:
    pass

import policy_core as core  # noqa: E402  (must follow the secrets setup)

MAX_QUESTIONS_PER_SESSION = 20  # protects the free-tier API quota on the public site
BASE = Path(__file__).parent
POLICY_TEXT = {p.title: p.policy_text for p in core.POLICIES}

st.set_page_config(page_title="Company Policy Assistant", page_icon="📘", layout="wide")
st.title("Company Policy Assistant")
st.caption(f"Rules-based search vs. LLM without / with a vector index · {len(core.POLICIES)} policies")

# ---------------------------------------------------------------- live comparison

st.header("Ask a policy question")
examples = ["How many vacation days do I get per year?",
            "My grandmother passed away. How much time off can I take?",
            "Can I work from another country for a few weeks?",
            "Can I bring my dog to the office?"]
with st.form("ask"):
    question = st.text_input("Question", value=examples[1], max_chars=300)
    st.caption("Try: " + " · ".join(f"*{e}*" for e in examples if e != examples[1]))
    use_judge = st.checkbox("Also check each answer for unsupported claims with the LLM judge "
                            "(slower, 3 extra API calls)")
    submitted = st.form_submit_button("Compare all three approaches", type="primary")

st.session_state.setdefault("asked", 0)


def grounding_line(r: core.PolicyResult, verdict: tuple[bool, str] | None) -> str:
    if verdict is not None:
        unsupported, reason = verdict
        return f"{'⚠️ **Unsupported**' if unsupported else '✅ **Supported**'} by the policy database: {reason}"
    if r.policy_title is None:
        return "ℹ️ No policy cited: the approach says the database does not answer this."
    if not r.policy_exists:
        return "⚠️ **Cites a policy that does not exist in the database.**"
    return "ℹ️ Cited policy exists; compare the answer with the policy text above."


if submitted and question.strip():
    if st.session_state.asked >= MAX_QUESTIONS_PER_SESSION:
        st.warning("Question limit for this session reached (it protects the free API quota). "
                   "Reload the page to continue.")
    else:
        st.session_state.asked += 1
        judge = None
        if use_judge:
            from benchmark import judge
        cols = st.columns(3)
        for col, fn in zip(cols, core.APPROACHES.values()):
            with col, st.container(border=True):
                with st.spinner("Thinking…"):
                    try:
                        r = fn(question.strip())
                        verdict = judge(question, r) if judge else None
                    except Exception as e:  # show the failure in this column, keep the others
                        st.subheader(fn.__name__)
                        st.error(f"This approach failed: {e}")
                        continue
                st.subheader(r.approach)
                st.markdown(f"**Answer:** {r.answer}")
                if r.policy_title:
                    st.markdown(f"**Relevant policy:** {r.policy_title}")
                    st.caption(f"Policy text: {POLICY_TEXT.get(r.policy_title, 'not found in database')}")
                else:
                    st.markdown("**Relevant policy:** none found")
                m1, m2 = st.columns(2)
                m1.metric("Response time", f"{r.response_time_s:.2f} s" if r.response_time_s >= 0.01
                          else f"{r.response_time_s * 1000:.1f} ms")
                m2.metric("Tokens", f"{r.total_tokens:,}",
                          help=f"{r.input_tokens:,} input + {r.output_tokens:,} output")
                st.markdown(grounding_line(r, verdict))

# ---------------------------------------------------------------- benchmark + write-up

st.header("Benchmark: 24 test questions")
summary = pd.read_csv(BASE / "benchmark_summary.csv", index_col=0)
shown = summary.copy()
for c in shown.columns[:5]:
    shown[c] = (shown[c] * 100).round(0).astype(int).astype(str) + " %"
shown["Median response time (s)"] = summary["Median response time (s)"].map(
    lambda s: f"{s:.3f} s" if s < 0.1 else f"{s:.2f} s")
shown["Avg tokens per question"] = summary["Avg tokens per question"].round(0).astype(int).map("{:,}".format)
shown.index.name = "Approach"
st.table(shown.T)  # Table 1

for paragraph in (BASE / "writeup.md").read_text(encoding="utf-8").strip().split("\n\n"):
    st.markdown(paragraph)

with st.expander("All 72 benchmark answers with the judge's reasoning"):
    details = pd.read_csv(BASE / "benchmark_results.csv")
    st.dataframe(  # Table 2
        details[["question", "type", "approach", "policy_title", "expected_policy", "policy_correct",
                 "unsupported", "judge_reason", "answer", "response_time_s", "total_tokens"]],
        hide_index=True, use_container_width=True,
    )
