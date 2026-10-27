"""Rate limiting on the question endpoint."""
import html
import re

import pytest


@pytest.fixture
def limited_client(db, monkeypatch):
    """A client with the limiter ON (the shared `client` fixture disables it)."""
    import app
    monkeypatch.setattr(app, "DEFAULT_WATCHLIST", [])
    app.init_db()
    app.limiter.enabled = True
    app.limiter.reset()
    import helpers
    monkeypatch.setattr(helpers.QuestionClassifier, "classify",
                        lambda self, q: dict(helpers.DEFAULT_CLASSIFICATION))
    yield app.app.test_client()
    app.limiter.reset()
    app.limiter.enabled = False


def post(c, ip, q="q"):
    return c.post("/", data={"question": q}, environ_base={"REMOTE_ADDR": ip})


def test_eleventh_question_in_a_minute_is_refused(limited_client):
    codes = [post(limited_client, "7.7.7.1").status_code for _ in range(12)]
    assert codes[:10] == [200] * 10
    assert codes[10:] == [429, 429]


def test_refusal_is_rendered_in_the_page_not_a_bare_error(limited_client):
    for _ in range(11):
        r = post(limited_client, "7.7.7.2")
    assert r.status_code == 429
    body = re.search(r"<pre>(.*?)</pre>", r.get_data(as_text=True), re.S)
    assert body, "429 should still render the normal template"
    assert "asking faster" in html.unescape(body.group(1))


def test_limit_is_per_client_not_global(limited_client):
    for _ in range(11):
        post(limited_client, "7.7.7.3")
    assert post(limited_client, "7.7.7.4").status_code == 200


def test_page_loads_do_not_consume_the_budget(limited_client):
    """GET must be exempt or a refresh would spend a question."""
    for _ in range(20):
        assert limited_client.get("/", environ_base={"REMOTE_ADDR": "7.7.7.5"}).status_code == 200
    assert post(limited_client, "7.7.7.5").status_code == 200


def test_cheap_routes_are_not_limited(limited_client):
    for _ in range(15):
        assert limited_client.get("/watchlist",
                                  environ_base={"REMOTE_ADDR": "7.7.7.6"}).status_code == 200


def test_retry_after_header_is_present(limited_client):
    for _ in range(11):
        r = post(limited_client, "7.7.7.7")
    assert r.headers.get("Retry-After")
