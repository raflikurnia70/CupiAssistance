# DataSense AI

**AI-Powered Data Science Assistant**
*"Understand Data. Ask Questions. Get Insights."*

Final project for the Hacktiv8 program *"LLM-Based Tools & Gemini API Integration for Data Scientists"*.

---

## Overview

DataSense AI is a Streamlit application that helps Data Scientists and Data Analysts understand a
dataset and get AI-assisted guidance, without training or running any Machine Learning model. It
combines automated Pandas-based dataset profiling with an LLM chat assistant (Llama, served via
Groq) that is grounded both in the user's actual uploaded data and in a Retrieval-Augmented
Generation (RAG) knowledge base of core Data Science concepts.

## Problem Statement

Before any modeling work can begin, a Data Scientist needs to understand a new dataset: its shape,
its data quality issues, and what needs to be fixed before analysis. This "data understanding"
phase is repetitive, and junior analysts often don't know what to look for or how to interpret basic
Data Science terminology encountered along the way. Switching between a notebook, documentation,
and search engines slows this process down.

## Solution

DataSense AI provides a single workspace where a user can:

1. Upload a CSV dataset and instantly see an automated profiling report (shape, types, missing
   values, duplicates, potential ID columns, potential outliers, descriptive statistics).
2. Ask questions about that specific dataset in natural language and get answers grounded in the
   actual profiling numbers (not guesses).
3. Ask general Data Science questions (e.g. "What is overfitting?") and get answers grounded in a
   bundled knowledge base via RAG, instead of an unverified model guess.
4. Keep a running conversation with context memory, so follow-up questions ("which column has the
   most missing values?" → "customer_age") stay coherent.

## Features

- CSV upload with automated Pandas profiling (rows, columns, dtypes, missing values, duplicates,
  numeric/categorical split, potential ID columns, IQR-based outlier indication).
- AI chat assistant backed by Llama via the Groq API, using role-based + contextual prompting.
- RAG-powered knowledge base covering 12 core Data Science topics.
- Chat memory via `st.session_state` so follow-up questions retain context.
- Quick-start prompts, dataset context card, and status indicators (AI / RAG) in the chat UI.
- Modern, minimal, dark-navy/off-white "AI SaaS" styled UI with custom CSS (no default Streamlit look).
- Graceful error handling for missing API key, missing PDF, invalid/empty CSV, and API failures.

## Data Science Workflow

CSV Upload → Pandas Profiling (`src/data_analysis.py`) → structured profile dict → rendered as
dashboard (metrics, data types table, quality bars, `describe()` table, missing-values chart) →
same structured profile is serialized into a compact text block and injected into the LLM prompt
as grounded context (see Hallucination Control below). No ML model is trained; the goal is data
understanding, not prediction.

## AI Architecture

```
User question ─▶ Dataset profile context (if a dataset is loaded)
              ─▶ RAG-retrieved knowledge base context (top-k chunks)
              ─▶ System prompt (role + rules) + conversation history
              ─▶ ChatGroq (Llama) or ChatGoogleGenerativeAI (Gemini), picked via LLM_PROVIDER ─▶ Answer
```

Both contexts are always attached; the system prompt instructs the model to prioritize the dataset
profile for dataset-specific questions and the knowledge base for general Data Science questions,
and to explicitly say when information isn't available rather than guessing.

## RAG Architecture

```
data/data_science_knowledge.pdf
        │  PyPDFLoader (langchain_community)
        ▼
   Documents (per page)
        │  RecursiveCharacterTextSplitter (chunk_size=800, overlap=120)
        ▼
   Chunks
        │  HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)
        ▼
   FAISS vector index  (persisted to vector_store/faiss_index/)
        │  similarity_search(query, k=3)
        ▼
   Relevant chunks ─▶ injected into the LLM prompt as KNOWLEDGE BASE context
```

FAISS was chosen over Chroma because it is a pure file-based index with no background sqlite
service, which is more predictable when deploying to Streamlit Community Cloud. The index is built
automatically on first run if it doesn't exist yet, and reused (cached via `st.cache_resource`) on
subsequent reruns.

The bundled PDF (`data/data_science_knowledge.pdf`) is a **Demonstration Knowledge Base** written
for this project — it is not official documentation for any tool or company.

## Prompt Engineering

Defined in [src/prompts.py](src/prompts.py):

- **Role-based prompting** — a system prompt defines DataSense AI's role and 11 explicit behavior
  rules (answer in Bahasa Indonesia by default, stay concise, ground answers in provided context,
  never invent dataset numbers, state limitations explicitly, distinguish observation / interpretation
  / recommendation, etc).
- **Contextual prompting** — every user turn is wrapped with a `DATASET PROFILE` block (from Pandas
  profiling) and a `KNOWLEDGE BASE CONTEXT` block (from RAG retrieval) before being sent to the model.
- **Hallucination awareness** — the dataset context block is explicitly labeled "actual data, use
  these numbers only", and the system prompt forbids inventing dataset figures or claiming an
  analysis happened when it did not. When a dataset isn't loaded, the model is told so directly and
  instructed to state the limitation rather than guess.

## Model Configuration

DataSense AI supports two interchangeable LLM providers, selected via `LLM_PROVIDER` in `.env`
(see [src/llm.py](src/llm.py)). If `LLM_PROVIDER` is left blank, the provider is auto-detected from
whichever API key is present (Groq takes priority if both are set).

| Parameter | Value | Purpose |
|---|---|---|
| Model (Groq) | `llama-3.3-70b-versatile` (via `GROQ_MODEL`) | Llama model served by Groq |
| Model (Gemini) | `gemini-3.6-flash` (via `GEMINI_MODEL`) | Google Gemini model |
| Temperature | `0.3` | Controls response creativity/randomness |
| Max Output Tokens | `512` (Groq) / `2048` (Gemini) | Controls maximum response length |
| Top P | `0.9` | Controls nucleus sampling |

Gemini's current "flash" models reason internally before answering, and that reasoning is billed
against `max_output_tokens`; at 512 the budget was consistently exhausted mid-thought (`finish_reason:
MAX_TOKENS`, verified while testing), producing a truncated, garbled answer. 2048 was the smallest
value that reliably left room for both the reasoning and a complete answer.

These are fixed, documented defaults shown read-only on the Settings page — the API key can never be
changed or viewed from the UI.

## Technology Stack

- **UI**: Streamlit, custom CSS
- **Data**: Pandas
- **LLM**: Groq API (Llama models) via `langchain-groq`, or Google Gemini via `langchain-google-genai`
- **RAG**: LangChain (`langchain`, `langchain-community`), PyPDF, FAISS, Sentence-Transformers
- **Charts**: Plotly

## Project Structure

```
CupiAssistance/
├── app.py                  # Streamlit entry point / page router
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── data/
│   └── data_science_knowledge.pdf   # Demonstration Knowledge Base (RAG source)
├── src/
│   ├── __init__.py
│   ├── llm.py               # Groq / LangChain LLM wrapper + configuration
│   ├── rag.py                # PDF -> split -> embeddings -> FAISS -> retriever
│   ├── data_analysis.py      # Pandas dataset profiling
│   ├── prompts.py            # System prompt, contextual prompt templates
│   └── ui.py                 # Custom CSS + reusable UI components
├── screenshots/
│   ├── overview.png
│   ├── data-analysis.png
│   └── ai-chat.png
└── .streamlit/
    └── config.toml
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in the provider you have a key for — never commit `.env`.

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | No | `groq` or `gemini`. Auto-detected from whichever key below is set if left blank. |
| `GROQ_API_KEY` | If using Groq | API key from [console.groq.com](https://console.groq.com/keys) |
| `GROQ_MODEL` | No | Overrides the default Llama model (`llama-3.3-70b-versatile`) |
| `GEMINI_API_KEY` | If using Gemini | API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | No | Overrides the default Gemini model (`gemini-3.6-flash`) |

## Running Locally

```bash
streamlit run app.py
```

The first run downloads the embedding model and builds the FAISS index from the bundled PDF
(cached afterwards in `vector_store/`).

## Example Questions

Dataset-specific (requires a CSV uploaded first):
- "Berapa jumlah data?"
- "Kolom mana yang memiliki missing value?"
- "Apa masalah utama dataset ini?"
- "Kolom mana yang kemungkinan menjadi feature?"
- "Apa yang perlu dilakukan sebelum modeling?"

General Data Science (answered via RAG):
- "Apa perbedaan precision dan recall?"
- "Bagaimana cara menangani missing values?"
- "Apa itu overfitting?"

## Screenshots

See [screenshots/](screenshots/) — `overview.png`, `data-analysis.png`, `ai-chat.png`.

## Deployment

Ready for Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Create a new app on [share.streamlit.io](https://share.streamlit.io) pointing at `app.py`.
3. Add `GROQ_API_KEY` (and optionally `GROQ_MODEL`) under the app's Secrets.
4. Deploy — the FAISS index will build automatically on first boot.

## Limitations

- No Machine Learning model is trained; the app performs data understanding, not prediction.
- The bundled knowledge base is a small demonstration document (12 short chapters), not an
  exhaustive Data Science reference.
- Outlier and ID-column detection use simple heuristics (IQR, name/uniqueness hints) and are
  indicative, not authoritative.
- Chat memory lives only in `st.session_state` for the current browser session; it is not persisted
  to a database.
- Requires a valid `GROQ_API_KEY` for the AI Chat feature; dataset profiling and the Data Analysis
  page work without one.

## Learning Implementation

**Session 1 — LLM + Prompting + Configuration**: role-based and contextual prompting
(`src/prompts.py`), a documented LLM configuration (temperature / max tokens / top P) surfaced on
the Settings page, and explicit hallucination-awareness rules in the system prompt.

**Session 2 — RAG + Vector Database + LangChain + Llama**: a full LangChain RAG pipeline
(`src/rag.py`) — PDF loader, text splitter, Sentence-Transformer embeddings, FAISS vector database,
similarity-search retriever — feeding a Llama model served through the Groq API.

**Session 3 — Streamlit + Chatbot + Deployment**: a multi-page Streamlit app (`app.py`) with a
`st.chat_message` / `st.chat_input` chatbot UI, session-state chat memory, custom CSS styling, and
a `.streamlit/config.toml` + `requirements.txt` ready for Streamlit Community Cloud deployment.

## Author

Rafli Kurnia Nugroho
