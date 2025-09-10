# GC Submittal Completeness Checker (Streamlit MVP)

Single-file Streamlit app to extract submittal requirements from a spec PDF and verify a submittal package using an LLM.

## Run locally

1. Create a Python virtual environment and install deps:

```zsh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Add your OpenAI API key to `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
```

3. Run the app:

```zsh
streamlit run app.py
```

## Deploy

- Streamlit Community Cloud: push this repo to GitHub, then connect it on https://share.streamlit.io and add `OPENAI_API_KEY` in the app secrets.
- Hugging Face Spaces: create a Streamlit space, paste `app.py`, and add `OPENAI_API_KEY` to secrets.

## Notes

- This is an MVP. The app uses an LLM to parse free-text PDFs — results should be reviewed by a human.
- Do not commit `.streamlit/secrets.toml` to source control.
