"""Provider-chain behaviour: the fallback that keeps a dead tier from failing a request."""
import types

import pytest

import llm


def _client(behaviour):
    """Minimal stand-in for openai.OpenAI."""
    def create(**kwargs):
        if isinstance(behaviour, Exception):
            raise behaviour
        msg = types.SimpleNamespace(content=behaviour)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                     _kwargs=kwargs)
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


@pytest.fixture
def chain(monkeypatch):
    """Build a two-tier chain with scripted behaviour per tier."""
    def _build(primary, backup, providers="ollama,openai"):
        monkeypatch.setattr(llm, "LLM_PROVIDERS", providers)
        monkeypatch.setattr(llm, "_REGISTRY", {
            "ollama": lambda: llm.Provider("ollama", "small", "http://x/v1", "k", 5.0),
            "openai": lambda: llm.Provider("openai", "big", None, "k", 5.0),
            "groq": lambda: llm.Provider("groq", "g", "http://g/v1", None, 5.0),
        })
        clients = {"ollama": _client(primary), "openai": _client(backup)}
        monkeypatch.setattr(llm, "_client", lambda p: clients[p.name])
    return _build


def test_primary_answers_and_no_fallback_is_recorded(chain):
    chain(primary="hello", backup="should not be used")
    r = llm.chat([{"role": "user", "content": "hi"}])
    assert r.text == "hello" and r.provider == "ollama" and r.fallbacks == []


def test_dead_primary_falls_through_to_backup(chain):
    chain(primary=ConnectionError("ollama is not running"), backup="rescued")
    r = llm.chat([{"role": "user", "content": "hi"}])
    assert r.text == "rescued" and r.provider == "openai"
    assert [name for name, _ in r.fallbacks] == ["ollama"]
    assert "ConnectionError" in r.fallbacks[0][1]


def test_empty_completion_counts_as_a_failure(chain):
    """A provider that returns nothing must not be treated as a success."""
    chain(primary="   ", backup="real answer")
    r = llm.chat([{"role": "user", "content": "hi"}])
    assert r.provider == "openai"


def test_every_tier_down_raises_one_clear_error(chain):
    chain(primary=ConnectionError("down"), backup=RuntimeError("401"))
    with pytest.raises(llm.AllProvidersFailed) as e:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "ollama" in str(e.value) and "openai" in str(e.value)


def test_providers_without_credentials_are_skipped(chain):
    """groq has no key, so the chain must fall straight through to openai."""
    chain(primary="unused", backup="from openai", providers="groq,openai")
    assert llm.chat([{"role": "user", "content": "hi"}]).provider == "openai"


def test_no_configured_provider_is_an_explicit_error(chain):
    chain(primary="x", backup="y", providers="groq")
    with pytest.raises(llm.AllProvidersFailed, match="No LLM provider configured"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_json_mode_sets_the_response_format(chain, monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="{}"))])

    monkeypatch.setattr(llm, "LLM_PROVIDERS", "ollama")
    monkeypatch.setattr(llm, "_REGISTRY", {
        "ollama": lambda: llm.Provider("ollama", "small", "http://x/v1", "k", 5.0)})
    monkeypatch.setattr(llm, "_client", lambda p: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))))

    llm.chat([{"role": "user", "content": "hi"}], json_mode=True)
    assert captured["response_format"] == {"type": "json_object"}
