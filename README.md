# Company Policy Assistant

Answers company policy questions and identifies the relevant policy. It compares three approaches over the 98 policies in `company_policies.csv`:

| Approach | How it works |
|---|---|
| Rules-based search | TF-IDF keyword matching; returns the closest policy verbatim (no LLM, 0 tokens) |
| LLM without vector index | Gemini (`gemini-3.5-flash-lite`) gets the whole policy database in every prompt |
| LLM with vector index | Policies are embedded once with `gemini-embedding-001`; only the 3 most similar are sent to Gemini |

## Files

| File | Purpose |
|---|---|
| `policy_core.py` | The three approaches (shared by the website and the Slack bot) |
| `app.py` | Streamlit website: live side-by-side comparison + benchmark + write-up |
| `benchmark.py` | Runs `test_questions.csv` through all approaches; an LLM judge (`gemini-3.1-flash-lite`) flags unsupported answers |
| `writeup.md` | The two-paragraph comparison shown on the website |
| `policy_embeddings.npy` | Cached vector index (rebuilt automatically if the CSV changes) |

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your Gemini API key in .env
streamlit run app.py
python benchmark.py         # optional: re-run the benchmark (~6 min on the free tier)
```

## Deploy (Streamlit Community Cloud)

Create an app from this repository with `app.py` as the main file. Under **Advanced settings → Secrets** add:

```toml
GEMINI_API_KEY = "your-key"
```
