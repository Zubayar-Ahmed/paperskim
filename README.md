# PaperSkim

A research paper summarizer with a modern UI. Paste a URL or upload a PDF, pick an AI provider + model, and get a structured summary. Ask follow-up questions about the paper, and inspect exactly what the model saw.

## Features

- **Multi-provider**: Ollama (local), Anthropic Claude, or OpenAI — pick per session
- **URL or PDF**: arXiv links auto-normalize to PDF; any HTML paper page works too
- **Structured summary**: TL;DR, Problem, Method, Key Results, Limitations, Why it matters
- **Chat with the paper**: grounded follow-up Q&A after the summary
- **Extraction transparency**: view the raw text, search it, see what was truncated

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deployed on Streamlit Community Cloud. Anthropic/OpenAI keys are bring-your-own via the sidebar (or set via Streamlit secrets). Ollama requires a local server, so that option only works when running locally.
