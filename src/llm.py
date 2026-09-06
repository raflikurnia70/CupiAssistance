"""LLM integration: pluggable between Llama models via Groq and Google Gemini,
both through LangChain.

Configuration (temperature / max tokens / top_p) is intentionally fixed to
sane, documented defaults rather than exposed as freeform user controls -
Settings page only *displays* these values (see README, section 15).

Provider selection (see .env.example):
- LLM_PROVIDER=groq|gemini picks explicitly.
- If unset, the provider is auto-detected from whichever API key is present
  (GROQ_API_KEY -> groq, GEMINI_API_KEY/GOOGLE_API_KEY -> gemini). Groq wins
  if both are set. Defaults to "groq" if neither is configured yet, so the
  Settings page has something sensible to display before setup.
"""

from __future__ import annotations

import os

PROVIDER_GROQ = "groq"
PROVIDER_GEMINI = "gemini"

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOP_P = 0.9

# Gemini's current "flash" models reason internally before answering, and
# that reasoning is billed against max_output_tokens. At 512 the budget is
# often exhausted mid-thought, producing a truncated/garbled answer (finish
# reason MAX_TOKENS). 2048 leaves enough room for reasoning + a full answer.
GEMINI_MAX_TOKENS = 2048


class LLMConfigError(Exception):
    """Raised when the LLM cannot be configured (e.g. missing API key)."""


class LLMResponseError(Exception):
    """Raised when the LLM call itself fails."""


def get_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in (PROVIDER_GROQ, PROVIDER_GEMINI):
        return explicit
    if os.getenv("GROQ_API_KEY"):
        return PROVIDER_GROQ
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return PROVIDER_GEMINI
    return PROVIDER_GROQ


def get_model_name() -> str:
    if get_provider() == PROVIDER_GEMINI:
        return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)


def get_provider_display_name() -> str:
    """Short label for the sidebar, e.g. 'Gemini' or 'Llama via Groq'."""
    return "Gemini" if get_provider() == PROVIDER_GEMINI else "Llama via Groq"


def get_max_tokens() -> int:
    return GEMINI_MAX_TOKENS if get_provider() == PROVIDER_GEMINI else DEFAULT_MAX_TOKENS


def get_provider_label() -> str:
    """Longer label for the Settings page, e.g. 'gemini-2.5-flash (Gemini)'."""
    return f"{get_model_name()} ({get_provider_display_name()})"


def is_configured() -> bool:
    if get_provider() == PROVIDER_GEMINI:
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    return bool(os.getenv("GROQ_API_KEY"))


def get_llm():
    provider = get_provider()

    if provider == PROVIDER_GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMConfigError(
                "GEMINI_API_KEY belum diatur. Tambahkan API key pada file .env (lihat .env.example)."
            )
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=get_model_name(),
            temperature=DEFAULT_TEMPERATURE,
            max_output_tokens=get_max_tokens(),
            top_p=DEFAULT_TOP_P,
        )

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
        max_tokens=get_max_tokens(),
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
