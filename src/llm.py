"""LLM integration: Llama models served via the Groq API, through LangChain.

Configuration (temperature / max tokens / top_p) is intentionally fixed to
sane, documented defaults rather than exposed as freeform user controls -
Settings page only *displays* these values (see README, section 15).
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOP_P = 0.9


class LLMConfigError(Exception):
    """Raised when the LLM cannot be configured (e.g. missing API key)."""


class LLMResponseError(Exception):
    """Raised when the LLM call itself fails."""


def get_model_name() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL)


def is_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def get_llm():
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise LLMConfigError(
            "GROQ_API_KEY belum diatur. Tambahkan API key pada file .env (lihat .env.example)."
        )
    return ChatGroq(
        api_key=api_key,
        model=get_model_name(),
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        model_kwargs={"top_p": DEFAULT_TOP_P},
    )


def _build_messages(system_prompt: str, history: list[dict], user_prompt: str):
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_prompt))
    return messages


def generate_response(system_prompt: str, user_prompt: str, history: list[dict] | None = None) -> str:
    """Call the LLM with role-based system prompt + contextual user prompt.

    `history` is prior chat turns (excluding the current question) so the
    model retains conversational context (see chat memory, section 18).
    """
    history = history or []
    llm = get_llm()
    messages = _build_messages(system_prompt, history, user_prompt)
    try:
        result = llm.invoke(messages)
    except Exception as exc:  # network/auth/rate-limit errors from the API
        raise LLMResponseError(f"Gagal mendapatkan respons dari model AI: {exc}") from exc

    content = (result.content or "").strip()
    if not content:
        raise LLMResponseError("Model AI mengembalikan respons kosong.")
    return content
