"""Intent routing: the deterministic dispatch between the classifier and the tools."""
import html
import re

import pytest

import app as A


def answer(client, question="q", ip="1.2.3.4"):
    body = client.post("/", data={"question": question},
                       environ_base={"REMOTE_ADDR": ip}).get_data(as_text=True)
    m = re.search(r"<pre>(.*?)</pre>", body, re.S)
    return html.unescape(m.group(1)).strip() if m else ""


def test_unknown_intent_explains_itself_instead_of_rendering_blank(client, stub_classify):
    """'other' had no branch, so the page came back with an empty answer."""
    stub_classify(intent="other")
    out = answer(client)
    assert out, "blank page regression"
    assert "couldn't match" in out and "rank a portfolio" in out


def test_an_intent_the_classifier_invented_also_falls_through(client, stub_classify):
    stub_classify(intent="definitely_not_a_real_intent")
    assert "couldn't match" in answer(client)


def test_stock_trend_without_a_symbol_asks_for_one(client, stub_classify):
    stub_classify(intent="stock_trend", symbol=None)
    assert "which stock" in answer(client).lower()


def test_should_buy_without_a_target_asks_for_one(client, stub_classify):
    stub_classify(intent="should_buy_stock", new_stock=None)
    assert "specify the stock" in answer(client).lower()


def test_news_topic_reaches_both_news_sources(client, stub_classify, monkeypatch):
    """The classifier extracted a topic and the pipeline threw it away."""
    seen = {}
    monkeypatch.setattr(A, "fetch_newsapi", lambda topic="general": seen.setdefault("topic", topic) or ["x"])
    monkeypatch.setattr(A, "get_finnhub_news", lambda category="general", max_articles=5: seen.setdefault("cat", category) or [])
    monkeypatch.setattr(A, "fetch_reddit_posts", lambda *a, **k: [])
    monkeypatch.setattr(A, "ask_llm", lambda q, c: "stub")
    monkeypatch.setattr(A, "ask_openai", lambda q, c: "stub")

    stub_classify(intent="get_finance_news", news_topic="crypto")
    answer(client)
    assert seen["topic"] == "crypto"
    assert seen["cat"] == "crypto"          # valid Finnhub category, passed through


def test_freeform_topic_falls_back_to_a_valid_finnhub_category(client, stub_classify, monkeypatch):
    seen = {}
    monkeypatch.setattr(A, "fetch_newsapi", lambda topic="general": seen.setdefault("topic", topic) or ["x"])
    monkeypatch.setattr(A, "get_finnhub_news", lambda category="general", max_articles=5: seen.setdefault("cat", category) or [])
    monkeypatch.setattr(A, "fetch_reddit_posts", lambda *a, **k: [])
    monkeypatch.setattr(A, "ask_llm", lambda q, c: "stub")
    monkeypatch.setattr(A, "ask_openai", lambda q, c: "stub")

    stub_classify(intent="get_finance_news", news_topic="banking layoffs")
    answer(client)
    assert seen["topic"] == "banking layoffs"   # verbatim to NewsAPI
    assert seen["cat"] == "general"             # Finnhub only accepts its fixed set


def test_classifier_failure_is_handled_not_a_500(client, monkeypatch):
    """classify() used to sit outside the try block, so any API error was a 500."""
    import helpers
    def boom(self, q):
        raise RuntimeError("provider unreachable")
    monkeypatch.setattr(helpers.QuestionClassifier, "classify", boom)

    r = client.post("/", data={"question": "anything"},
                    environ_base={"REMOTE_ADDR": "1.2.3.4"})
    assert r.status_code == 200
    assert "provider unreachable" in r.get_data(as_text=True)


def test_empty_question_is_rejected_before_any_work(client):
    assert "Please ask a question" in answer(client, question="   ")


# ---------- graceful degradation of the news path ----------

def _news_stubs(monkeypatch, newsapi=None, reddit=None, finnhub=None):
    from sources import SourceError
    def mk(val):
        if isinstance(val, Exception):
            return lambda *a, **k: (_ for _ in ()).throw(val)
        return lambda *a, **k: (val if val is not None else [])
    monkeypatch.setattr(A, "fetch_newsapi", mk(newsapi))
    monkeypatch.setattr(A, "fetch_reddit_posts", mk(reddit))
    monkeypatch.setattr(A, "get_finnhub_news", mk(finnhub))
    monkeypatch.setattr(A, "ask_llm", lambda q, c: f"SYNTH[{len(c.split(chr(10)+chr(10)))} blocks]")
    import sources
    monkeypatch.setattr(sources, "BACKOFF_SECONDS", 0)
    return SourceError


def test_one_dead_news_source_still_produces_an_answer(client, stub_classify, monkeypatch):
    """This used to lose the whole answer: fetch_newsapi raised and nothing caught it."""
    E = _news_stubs(monkeypatch, newsapi=Exception("NewsAPI 429"),
                    reddit=["r1"], finnhub=["f1"])
    stub_classify(intent="get_finance_news", news_topic="general")
    out = answer(client)
    assert "SYNTH" in out, "a dead source must not take the answer down"
    assert "unavailable: NewsAPI" in out


def test_the_answer_states_which_sources_it_used(client, stub_classify, monkeypatch):
    """A degraded answer that looks complete is the failure worth avoiding."""
    _news_stubs(monkeypatch, newsapi=["n1"], reddit=["r1"], finnhub=["f1"])
    stub_classify(intent="get_finance_news", news_topic="general")
    out = answer(client)
    assert "Sources used: NewsAPI, Reddit, Finnhub" in out


def test_all_sources_down_refuses_instead_of_improvising(client, stub_classify, monkeypatch):
    """An empty context must never reach the model, or it answers from memory."""
    called = {"synth": False}
    _news_stubs(monkeypatch, newsapi=Exception("down"), reddit=Exception("down"),
                finnhub=Exception("down"))
    monkeypatch.setattr(A, "ask_llm", lambda q, c: called.__setitem__("synth", True) or "LEAK")
    stub_classify(intent="get_finance_news", news_topic="general")
    out = answer(client)
    assert not called["synth"], "the model must not be asked to answer with no context"
    assert "No news sources are reachable" in out
