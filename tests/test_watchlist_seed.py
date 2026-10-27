"""DEFAULT_WATCHLIST seeding: populate a fresh DB, never clobber user edits."""
import sqlite3

import pytest

SEED = [("IVV", 0.6, 5.0), ("BTC-USD", 0.4, 10.0)]


def rows(path):
    conn = sqlite3.connect(path)
    out = conn.execute("SELECT ticker, weight, alert_threshold FROM portfolios "
                       "ORDER BY ticker").fetchall()
    conn.close()
    return out


@pytest.fixture
def seeded(db, monkeypatch):
    import app
    monkeypatch.setattr(app, "DEFAULT_WATCHLIST", SEED)
    app.init_db()
    return app, db


def test_fresh_database_is_seeded(seeded):
    _, path = seeded
    assert rows(path) == [("BTC-USD", 0.4, 10.0), ("IVV", 0.6, 5.0)]


def test_restart_does_not_duplicate(seeded):
    app, path = seeded
    app.init_db()
    app.init_db()
    assert len(rows(path)) == 2


def test_user_edits_survive_a_restart(seeded):
    """Deleting a ticker must not be undone by the next boot."""
    app, path = seeded
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM portfolios WHERE ticker='IVV'")
    conn.execute("UPDATE portfolios SET weight=0.99 WHERE ticker='BTC-USD'")
    conn.commit(); conn.close()

    app.init_db()
    assert rows(path) == [("BTC-USD", 0.99, 10.0)]


def test_empty_seed_leaves_the_table_alone(db, monkeypatch):
    import app
    monkeypatch.setattr(app, "DEFAULT_WATCHLIST", [])
    app.init_db()
    assert rows(db) == []


def test_seeded_weights_sum_to_one():
    """A guard on the shipped default, not on the seeding mechanism."""
    import app
    total = sum(w for _, w, _ in app.DEFAULT_WATCHLIST)
    assert total == pytest.approx(1.0), f"shipped watchlist sums to {total}"
