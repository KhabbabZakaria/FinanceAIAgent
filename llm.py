"""Single entry point for every LLM call in the app.

Each provider speaks the OpenAI chat-completions API, so one client type covers a
local Ollama model and hosted OpenAI alike -- only base_url, key and model differ.
Calls walk the provider chain in order and fall through on failure, so a stopped
Ollama server or a dead API key degrades to the next tier instead of failing the
request.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import time

from openai import OpenAI

from config import (
    GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL,
    LLM_PROVIDERS, OLLAMA_BASE_URL, OLLAMA_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
)

logger = logging.getLogger("financeai.llm")


class AllProvidersFailed(RuntimeError):
    """Every provider in the chain errored or was unreachable."""


@dataclass
class Provider:
    name: str
    model: str
    base_url: Optional[str]
    api_key: Optional[str]
    timeout: float

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    latency_ms: int
    # (provider, error) for each tier that failed before this one answered
    fallbacks: List[Tuple[str, str]] = field(default_factory=list)


# A local model needs no credential, but the OpenAI SDK requires a non-empty key,
# hence the "ollama" sentinel. Adding a tier is a registry entry plus an env var.
_REGISTRY = {
    "ollama": lambda: Provider("ollama", OLLAMA_MODEL, OLLAMA_BASE_URL, "ollama", 120.0),
    "openai": lambda: Provider("openai", OPENAI_MODEL, None, OPENAI_API_KEY, 60.0),
    "groq": lambda: Provider("groq", GROQ_MODEL, GROQ_BASE_URL, GROQ_API_KEY, 60.0),
}

_clients: Dict[str, OpenAI] = {}


def _chain() -> List[Provider]:
    """Providers named in LLM_PROVIDERS, in order, skipping unconfigured ones."""
    chain = []
    for name in (p.strip().lower() for p in LLM_PROVIDERS.split(",")):
        if not name:
            continue
        if name not in _REGISTRY:
            logger.warning("Unknown LLM provider %r in LLM_PROVIDERS; skipping", name)
            continue
        provider = _REGISTRY[name]()
        if not provider.configured:
            logger.info("Provider %s has no credential configured; skipping", name)
            continue
        chain.append(provider)
    return chain


def _client(provider: Provider) -> OpenAI:
    if provider.name not in _clients:
        # max_retries=0: the chain handles failover, so fall through to the next
        # tier immediately rather than retrying a provider that is already down.
        _clients[provider.name] = OpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            timeout=provider.timeout,
            max_retries=0,
        )
    return _clients[provider.name]


def chat(messages: List[Dict[str, str]], *, json_mode: bool = False,
         max_tokens: int = 512, temperature: float = 0.0) -> LLMResult:
    """Send `messages` to the first provider that answers.

    Set json_mode=True to constrain the reply to a JSON object. Raises
    AllProvidersFailed only when every configured tier has been tried.
    """
    chain = _chain()
    if not chain:
        raise AllProvidersFailed(
            f"No LLM provider configured (LLM_PROVIDERS={LLM_PROVIDERS!r}). "
            "Start one with `ollama serve`, or set OPENAI_API_KEY."
        )

    kwargs: Dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    fallbacks: List[Tuple[str, str]] = []
    for provider in chain:
        started = time.monotonic()
        try:
            response = _client(provider).chat.completions.create(model=provider.model, **kwargs)
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("provider returned an empty completion")
        except Exception as e:
            logger.warning("LLM provider %s (%s) failed: %s", provider.name, provider.model, e)
            fallbacks.append((provider.name, f"{type(e).__name__}: {e}"))
            continue

        latency_ms = int((time.monotonic() - started) * 1000)
        if fallbacks:
            logger.info("Request served by %s after %d fallback(s)", provider.name, len(fallbacks))
        return LLMResult(text, provider.name, provider.model, latency_ms, fallbacks)

    raise AllProvidersFailed(
        "All LLM providers failed: " + "; ".join(f"{n} ({e})" for n, e in fallbacks)
    )


def health() -> List[Dict[str, Any]]:
    """Per-provider reachability, for a status page or a smoke test."""
    report = []
    for provider in _chain():
        started = time.monotonic()
        entry: Dict[str, Any] = {"provider": provider.name, "model": provider.model}
        try:
            _client(provider).chat.completions.create(
                model=provider.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            entry["ok"] = True
        except Exception as e:
            entry["ok"] = False
            entry["error"] = f"{type(e).__name__}: {e}"
        entry["latency_ms"] = int((time.monotonic() - started) * 1000)
        report.append(entry)
    return report
