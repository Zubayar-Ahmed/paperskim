import io
import json
import os
import re
import time
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader

OLLAMA_BASE = "http://localhost:11434"

SUMMARY_SYSTEM = """You are summarizing an academic research paper for a reader who wants to understand the work in under two minutes. Produce a structured markdown summary using this exact format:

## TL;DR
One sentence in plain language.

## Problem
What problem does the paper tackle, and why does it matter?

## Method
The core approach or contribution. Be concrete — name the technique.

## Key Results
2-4 bullets with the most important findings. Include numbers when the paper gives them.

## Limitations
Caveats the authors mention, or gaps you notice.

## Why it matters
Who should care about this work and why.

Paper text:
---
{text}
---
"""

CHAT_SYSTEM = """You are a research assistant helping a reader understand a paper. Answer questions strictly based on the paper text below. If the paper does not cover something, say so — do not speculate beyond what's written. Keep answers focused and cite specific sections or phrases from the paper when relevant.

Paper text:
---
{text}
---
"""


# --- Text extraction ----------------------------------------------------------


def extract_pdf_text(file_or_bytes) -> str:
    if isinstance(file_or_bytes, (bytes, bytearray)):
        file_or_bytes = io.BytesIO(file_or_bytes)
    reader = PdfReader(file_or_bytes)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def extract_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def normalize_arxiv_url(url: str) -> str:
    m = re.match(r"https?://(?:www\.)?arxiv\.org/abs/([\w\.\-/]+)", url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
    return url


def fetch_paper_text(url: str) -> tuple[str, str]:
    url = normalize_arxiv_url(url.strip())
    headers = {"User-Agent": "Mozilla/5.0 (paper-summarizer)"}
    r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        return extract_pdf_text(r.content), "pdf"
    return extract_html_text(r.text), "html"


# --- Providers ----------------------------------------------------------------


class OllamaProvider:
    name = "Ollama (local)"
    needs_key = False

    def list_models(self, _key=None):
        try:
            r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except requests.RequestException:
            return []

    def stream_chat(self, model, system, messages, _key=None):
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": True,
        }
        with requests.post(
            f"{OLLAMA_BASE}/api/chat", json=payload, stream=True, timeout=600
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content")
                if content:
                    yield content
                if chunk.get("done"):
                    break


class AnthropicProvider:
    name = "Anthropic Claude"
    needs_key = True

    def list_models(self, _key=None):
        return [
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ]

    def stream_chat(self, model, system, messages, key):
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 2000,
            "stream": True,
            "system": system,
            "messages": messages,
        }
        with requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            stream=True,
            timeout=600,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")


class OpenAIProvider:
    name = "OpenAI"
    needs_key = True

    def list_models(self, _key=None):
        return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]

    def stream_chat(self, model, system, messages, key):
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "stream": True,
            "messages": [{"role": "system", "content": system}] + messages,
        }
        with requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=600,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    break
                try:
                    event = json.loads(data)
                    delta = event["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                content = delta.get("content")
                if content:
                    yield content


PROVIDERS = {
    "Ollama (local)": OllamaProvider(),
    "Anthropic Claude": AnthropicProvider(),
    "OpenAI": OpenAIProvider(),
}

ENV_KEY_NAMES = {
    "Anthropic Claude": "ANTHROPIC_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
}


# --- Styling ------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.hero {
    padding: 24px 0 8px 0;
    margin-bottom: 8px;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(120deg, #7c3aed 0%, #ec4899 50%, #f59e0b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1.05rem;
    opacity: 0.65;
    margin-top: 6px;
    font-weight: 400;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid rgba(128,128,128,0.18);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    height: auto;
    padding: 10px 18px;
    border-radius: 10px 10px 0 0;
    font-weight: 500;
    font-size: 0.95rem;
    background: transparent;
    transition: background 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(124,58,237,0.08);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(124,58,237,0.12) 0%, rgba(124,58,237,0.04) 100%) !important;
    color: #7c3aed !important;
    font-weight: 600 !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #7c3aed 0%, #ec4899 100%);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 22px;
    transition: transform 0.1s ease, box-shadow 0.15s ease;
    box-shadow: 0 2px 8px rgba(124,58,237,0.25);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(124,58,237,0.35);
}

section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(124,58,237,0.12);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

.meta-card {
    padding: 14px 18px;
    border-radius: 12px;
    background: linear-gradient(120deg, rgba(124,58,237,0.06) 0%, rgba(236,72,153,0.04) 100%);
    border: 1px solid rgba(124,58,237,0.15);
    margin: 4px 0 16px 0;
    font-size: 0.92rem;
}
.meta-card b { color: #7c3aed; }

.stat-row { display: flex; gap: 24px; flex-wrap: wrap; margin: 8px 0 20px 0; }
.stat { flex: 1; min-width: 110px; padding: 14px 18px; border-radius: 12px;
        background: rgba(124,58,237,0.04); border: 1px solid rgba(124,58,237,0.12); }
.stat-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.6; font-weight: 600; }
.stat-value { font-size: 1.45rem; font-weight: 700; margin-top: 2px; }

div[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 4px 8px;
}
</style>
"""


# --- State helpers ------------------------------------------------------------


def init_state():
    defaults = {
        "paper_text": "",
        "source_label": "",
        "summary": "",
        "summary_elapsed": 0.0,
        "messages": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def reset_paper():
    st.session_state.paper_text = ""
    st.session_state.source_label = ""
    st.session_state.summary = ""
    st.session_state.summary_elapsed = 0.0
    st.session_state.messages = []


# --- UI -----------------------------------------------------------------------

st.set_page_config(
    page_title="PaperSkim",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
init_state()

st.markdown(
    """
    <div class="hero">
      <div class="hero-title">PaperSkim</div>
      <div class="hero-sub">Paste a URL or drop a PDF — get a structured summary, ask follow-up questions, and inspect exactly what the model saw.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Model")
    provider_name = st.selectbox("Provider", list(PROVIDERS.keys()), key="provider_name")
    provider = PROVIDERS[provider_name]

    api_key = None
    if provider.needs_key:
        env_name = ENV_KEY_NAMES.get(provider_name, "")
        env_val = os.environ.get(env_name, "")
        api_key = st.text_input(
            f"{provider_name} API key",
            value=env_val,
            type="password",
            help=f"Or set ${env_name} in your environment.",
            key=f"key_{provider_name}",
        )

    models = provider.list_models(api_key)
    if not models:
        if isinstance(provider, OllamaProvider):
            st.error(
                "Ollama not reachable.\n\n"
                "1. Install: https://ollama.com\n"
                "2. `ollama pull llama3.1`\n"
                "3. `ollama serve`"
            )
        else:
            st.warning("No models available.")
        model = None
    else:
        model = st.selectbox("Model", models, key=f"model_{provider_name}")

    st.markdown("### 📏 Context")
    max_chars = st.slider(
        "Max chars sent to model",
        min_value=5000,
        max_value=120000,
        value=40000,
        step=5000,
        help="Longer = more context but slower. Cloud models handle 80k+ fine; local models usually want ≤25k.",
    )

    st.divider()
    if st.button("🔄 Reset paper & chat", use_container_width=True):
        reset_paper()
        st.rerun()


tab_load, tab_summary, tab_ask, tab_extract = st.tabs(
    ["📥 Load Paper", "✨ Summary", "💬 Ask Questions", "🔍 Extracted Text"]
)


# --- Tab: Load ----------------------------------------------------------------

with tab_load:
    st.markdown("#### Load a paper")
    source = st.radio("Source", ["URL", "Upload PDF"], horizontal=True, label_visibility="collapsed")

    if source == "URL":
        col1, col2 = st.columns([5, 1])
        with col1:
            url = st.text_input(
                "Paper URL",
                placeholder="https://arxiv.org/abs/2301.00001",
                label_visibility="collapsed",
            )
        with col2:
            fetch_clicked = st.button("Fetch", type="primary", use_container_width=True)

        if fetch_clicked and url:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                st.error("URL must start with http:// or https://")
            else:
                try:
                    with st.spinner(f"Fetching {parsed.netloc}..."):
                        text, kind = fetch_paper_text(url)
                    st.session_state.paper_text = text
                    st.session_state.source_label = f"{kind.upper()} · {parsed.netloc}"
                    st.session_state.summary = ""
                    st.session_state.messages = []
                    st.success("Paper loaded. Head over to **✨ Summary** or **💬 Ask Questions**.")
                except requests.RequestException as e:
                    st.error(f"Fetch failed: {e}")
    else:
        uploaded = st.file_uploader("Upload paper PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded:
            with st.spinner("Extracting text from PDF..."):
                text = extract_pdf_text(uploaded)
            st.session_state.paper_text = text
            st.session_state.source_label = f"PDF · {uploaded.name}"
            st.session_state.summary = ""
            st.session_state.messages = []
            st.success("Paper loaded. Head over to **✨ Summary** or **💬 Ask Questions**.")

    if st.session_state.paper_text:
        text = st.session_state.paper_text
        words = len(text.split())
        sent_chars = min(len(text), max_chars)
        st.markdown(
            f"""
            <div class="stat-row">
              <div class="stat"><div class="stat-label">Source</div><div class="stat-value" style="font-size:1rem;">{st.session_state.source_label}</div></div>
              <div class="stat"><div class="stat-label">Words</div><div class="stat-value">{words:,}</div></div>
              <div class="stat"><div class="stat-label">Characters</div><div class="stat-value">{len(text):,}</div></div>
              <div class="stat"><div class="stat-label">Sent to model</div><div class="stat-value">{sent_chars:,}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- Tab: Summary -------------------------------------------------------------


def ready_to_run():
    return (
        bool(st.session_state.paper_text)
        and model is not None
        and (not provider.needs_key or api_key)
    )


def readiness_warning():
    if not st.session_state.paper_text:
        st.info("Load a paper in the **📥 Load Paper** tab first.")
    elif provider.needs_key and not api_key:
        st.warning(f"Enter your {provider_name} API key in the sidebar.")
    elif not model:
        st.warning("Pick a model in the sidebar.")


with tab_summary:
    st.markdown("#### Structured summary")

    if not ready_to_run():
        readiness_warning()
    else:
        col1, col2 = st.columns([1, 5])
        with col1:
            generate = st.button(
                "Generate" if not st.session_state.summary else "Regenerate",
                type="primary",
                use_container_width=True,
            )
        with col2:
            st.markdown(
                f"<div class='meta-card'>Using <b>{model}</b> via <b>{provider_name}</b> · "
                f"{min(len(st.session_state.paper_text), max_chars):,} chars in context</div>",
                unsafe_allow_html=True,
            )

        if generate:
            placeholder = st.empty()
            output = ""
            start = time.time()
            try:
                system = SUMMARY_SYSTEM.format(text=st.session_state.paper_text[:max_chars])
                messages = [{"role": "user", "content": "Please produce the structured summary now."}]
                for chunk in provider.stream_chat(model, system, messages, api_key):
                    output += chunk
                    placeholder.markdown(output)
                st.session_state.summary = output
                st.session_state.summary_elapsed = time.time() - start
            except requests.RequestException as e:
                st.error(f"Summarization failed: {e}")

        if st.session_state.summary:
            if not generate:
                st.markdown(st.session_state.summary)
            if st.session_state.summary_elapsed:
                st.caption(f"Generated in {st.session_state.summary_elapsed:.1f}s")
            st.download_button(
                "⬇ Download summary (markdown)",
                data=st.session_state.summary,
                file_name="summary.md",
                mime="text/markdown",
            )


# --- Tab: Ask -----------------------------------------------------------------

with tab_ask:
    st.markdown("#### Ask follow-up questions about the paper")

    if not ready_to_run():
        readiness_warning()
    else:
        st.markdown(
            f"<div class='meta-card'>Chatting with the paper using <b>{model}</b> via <b>{provider_name}</b>. Answers are grounded in the extracted text only.</div>",
            unsafe_allow_html=True,
        )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_q = st.chat_input("Ask anything about this paper…")
        if user_q:
            st.session_state.messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                output = ""
                try:
                    system = CHAT_SYSTEM.format(text=st.session_state.paper_text[:max_chars])
                    for chunk in provider.stream_chat(
                        model, system, st.session_state.messages, api_key
                    ):
                        output += chunk
                        placeholder.markdown(output + "▍")
                    placeholder.markdown(output)
                    st.session_state.messages.append({"role": "assistant", "content": output})
                except requests.RequestException as e:
                    st.error(f"Chat failed: {e}")

        if st.session_state.messages:
            if st.button("🗑 Clear conversation"):
                st.session_state.messages = []
                st.rerun()


# --- Tab: Extracted text ------------------------------------------------------

with tab_extract:
    st.markdown("#### Exactly what was extracted")
    st.caption("Inspect the raw text pulled from the paper — this is what the model sees.")

    if not st.session_state.paper_text:
        st.info("Load a paper in the **📥 Load Paper** tab first.")
    else:
        text = st.session_state.paper_text
        sent_chars = min(len(text), max_chars)
        truncated = len(text) > max_chars

        col1, col2, col3 = st.columns(3)
        col1.metric("Total chars", f"{len(text):,}")
        col2.metric("Words", f"{len(text.split()):,}")
        col3.metric("Sent to model", f"{sent_chars:,}", delta=f"-{len(text)-sent_chars:,}" if truncated else None)

        search = st.text_input("🔎 Search in extracted text", placeholder="keyword or phrase")
        display_text = text
        if search:
            hits = [m.start() for m in re.finditer(re.escape(search), text, re.IGNORECASE)]
            st.caption(f"Found **{len(hits)}** match{'es' if len(hits) != 1 else ''} for '{search}'.")

        view = st.radio(
            "View",
            ["Full extracted text", "Only what's sent to model", "Only what's truncated"],
            horizontal=True,
        )
        if view == "Full extracted text":
            display_text = text
        elif view == "Only what's sent to model":
            display_text = text[:max_chars]
        else:
            display_text = text[max_chars:] if truncated else "(Nothing was truncated — full text was sent.)"

        st.text_area(
            "Extracted text",
            value=display_text,
            height=500,
            label_visibility="collapsed",
        )

        st.download_button(
            "⬇ Download full extracted text",
            data=text,
            file_name="extracted.txt",
            mime="text/plain",
        )
