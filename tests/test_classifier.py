"""Pure-function tests for the question classifier.

No model is called: these cover the normalisation and parsing that sit either
side of the LLM, which is where the router's inputs actually get shaped.
"""
import pytest

from helpers import QuestionClassifier as Q


@pytest.mark.parametrize("raw,expected", [
    (["50%", "30%", "20%"], [0.5, 0.3, 0.2]),
    ([0.5, 0.3, 0.2], [0.5, 0.3, 0.2]),
    ([50, 30, 20], [0.5, 0.3, 0.2]),
    ([60, 40], [0.6, 0.4]),
])
def test_weights_normalise_to_decimals_summing_to_one(raw, expected):
    got = Q._normalize_weights(raw)
    assert got == pytest.approx(expected, abs=1e-9)
    assert sum(got) == pytest.approx(1.0)


def test_unparseable_weight_becomes_zero_not_a_crash():
    assert Q._normalize_weights(["abc", 100]) == pytest.approx([0.0, 1.0])


def test_empty_weights_are_left_alone():
    assert Q._normalize_weights([]) == []


@pytest.mark.parametrize("raw,expected", [
    (" aapl ", "AAPL"), ("brk b", "BRKB"), ("msft", "MSFT"),
])
def test_tickers_are_normalised(raw, expected):
    assert Q._normalize_ticker(raw) == expected


def test_json_is_extracted_from_chatty_output():
    text = 'Sure! Here you go:\n```json\n{"intent":"stock_trend"}\n```\nHope that helps.'
    assert Q._try_extract_json(text) == '{"intent":"stock_trend"}'


def test_garbage_output_falls_back_to_a_safe_default(monkeypatch):
    """An unparseable reply must route to 'other', never raise."""
    import helpers
    monkeypatch.setattr(helpers, "chat", lambda *a, **k: _Result("not json at all"))
    out = Q().classify("whatever")
    assert out["intent"] == "other"
    assert out["tickers"] == [] and out["weights"] == []


class _Result:
    def __init__(self, text):
        self.text, self.provider, self.model, self.latency_ms = text, "stub", "stub", 1


# ---------- recovering the candidate stock ----------
#
# The few-shot examples once used AAPL, TSLA and MSFT as existing holdings, so
# the model learned a ticker->slot association and filed "should I buy AAPL"
# under `tickers`. The prompt no longer reuses a ticker across slots; this
# repair catches whatever still slips through.

def repair(question, **parsed):
    base = {"intent": "should_buy_stock", "tickers": [], "weights": [], "new_stock": None}
    return Q._repair_should_buy(question, {**base, **parsed})


def test_candidate_named_in_the_question_is_recovered():
    out = repair("should I buy AAPL? I hold IVV 60%, KO 40%",
                 tickers=["IVV", "KO"], weights=[0.6, 0.4])
    assert out["new_stock"] == "AAPL"
    assert out["tickers"] == ["IVV", "KO"], "holdings must be left alone"


def test_a_lone_misfiled_holding_is_promoted():
    """Nobody asks whether to buy a one-stock portfolio they already own."""
    out = repair("Give me the news about Apple (AAPL). should I invest here?",
                 tickers=["AAPL"], weights=[1.0])
    assert out["new_stock"] == "AAPL"
    assert out["tickers"] == [] and out["weights"] == []


def test_a_correct_classification_is_never_overwritten():
    out = repair("should I buy NVDA? I hold IVV 60%, KO 40%",
                 tickers=["IVV", "KO"], weights=[0.6, 0.4], new_stock="NVDA")
    assert out["new_stock"] == "NVDA"


def test_other_intents_are_untouched():
    out = repair("rank my portfolio: IVV 60%, KO 40%",
                 intent="rank_portfolio", tickers=["IVV", "KO"], weights=[0.6, 0.4])
    assert out["new_stock"] is None and out["tickers"] == ["IVV", "KO"]


def test_ambiguous_mentions_are_left_alone():
    """Two unexplained tickers: guessing which is the candidate would be worse
    than asking."""
    out = repair("should I buy AAPL or MSFT? I hold IVV 100%",
                 tickers=["IVV"], weights=[1.0])
    assert out["new_stock"] is None


def test_common_uppercase_words_are_not_mistaken_for_tickers():
    out = repair("should I buy an ETF for my IRA?", tickers=[], weights=[])
    assert out["new_stock"] is None


# ---------- recovering a holding the model dropped ----------
#
# Real failure: asked to rank 7 holdings the model returned 6, dropping
# BTC-USD. _normalize_weights then rescaled the survivors from 98% back to
# 100%, so a 30% holding was reported as 30.61% and the loss was invisible.

def drop_repair(question, **parsed):
    base = {"intent": "rank_portfolio", "tickers": [], "weights": [], "new_stock": None}
    return Q._repair_dropped_holdings(question, {**base, **parsed})


def test_a_dropped_hyphenated_ticker_is_restored():
    out = drop_repair("rank my portfolio: IVV 30%, PLTR 10%, BTC-USD 2%",
                      tickers=["IVV", "PLTR"], weights=[30.0, 10.0])
    assert out["tickers"] == ["IVV", "PLTR", "BTC-USD"]
    assert out["weights"] == [30.0, 10.0, 2.0]


def test_the_restored_weight_is_the_one_the_user_typed():
    out = drop_repair("IVV 30%, IWM 20%, BTC-USD 2%",
                      tickers=["IVV", "IWM"], weights=[30.0, 20.0])
    assert out["weights"][-1] == 2.0


def test_a_complete_classification_is_left_alone():
    out = drop_repair("rank my portfolio: IVV 60%, KO 40%",
                      tickers=["IVV", "KO"], weights=[60.0, 40.0])
    assert out["tickers"] == ["IVV", "KO"] and out["weights"] == [60.0, 40.0]


def test_the_candidate_stock_is_not_mistaken_for_a_holding():
    """"should I buy NVDA? I hold IVV 60%, KO 40%" must not add NVDA to holdings."""
    out = drop_repair("should I buy NVDA? I hold IVV 60%, KO 40%",
                      intent="should_buy_stock", tickers=["IVV", "KO"],
                      weights=[60.0, 40.0], new_stock="NVDA")
    assert "NVDA" not in out["tickers"]


def test_percentages_without_a_ticker_are_ignored():
    out = drop_repair("I hold 60% in cash", tickers=[], weights=[])
    assert out["tickers"] == []


def test_intents_without_holdings_are_untouched():
    out = drop_repair("what's going on with IVV 30%?", intent="stock_trend",
                      tickers=[], weights=[])
    assert out["tickers"] == []


def test_misaligned_lengths_are_left_alone():
    """If tickers and weights already disagree, appending would make it worse."""
    out = drop_repair("IVV 30%, BTC-USD 2%", tickers=["IVV", "KO"], weights=[30.0])
    assert out["tickers"] == ["IVV", "KO"]


# ---------- rejecting holdings the model invented ----------
#
# Real failure: "should I add PLTR to my portfolio" names no holdings, and the
# model returned the few-shot example's portfolio verbatim (VTI 60%, JNJ 40%).
# Recommending against a portfolio the user does not own is a silent wrong
# answer, which is worse than admitting we do not know what they hold.

def invented(question, **parsed):
    base = {"intent": "should_buy_stock", "tickers": [], "weights": [], "new_stock": None}
    return Q._drop_invented_holdings(question, {**base, **parsed})


def test_example_tickers_are_read_from_the_prompt():
    """Derived, not hardcoded, so editing the examples cannot desync this."""
    ex = Q._example_tickers()
    assert {"VTI", "JNJ", "IVV", "KO"} <= ex
    assert "AAPL" not in ex, "the examples must not reuse a popular ticker"


def test_holdings_copied_from_the_examples_are_dropped():
    out = invented("should I add PLTR to my portfolio",
                   tickers=["VTI", "JNJ"], weights=[0.6, 0.4], new_stock="PLTR")
    assert out["tickers"] == [] and out["weights"] == []


def test_the_same_tickers_are_kept_when_the_user_names_them():
    out = invented("should I buy COST? I hold VTI 60%, JNJ 40%",
                   tickers=["VTI", "JNJ"], weights=[0.6, 0.4], new_stock="COST")
    assert out["tickers"] == ["VTI", "JNJ"]


def test_holdings_outside_the_example_set_are_never_touched():
    """Only copy-contamination is suspect; a ticker the prompt never shows
    cannot have been copied from it."""
    out = invented("should I buy NVDA", tickers=["ZZZZ", "QQQQ"],
                   weights=[0.5, 0.5], new_stock="NVDA")
    assert out["tickers"] == ["ZZZZ", "QQQQ"]


def test_a_null_inside_the_ticker_list_does_not_crash():
    """The model emits null occasionally; it used to reach .upper() and raise.
    Stripping runs unconditionally, before any of the repairs."""
    out = Q._strip_null_tickers({"tickers": ["IVV", None], "weights": [0.3, 0.02]})
    assert out["tickers"] == ["IVV"] and out["weights"] == [0.3]


def test_null_stripping_keeps_tickers_and_weights_aligned():
    out = Q._strip_null_tickers({"tickers": [None, "KO", None], "weights": [1, 2, 3]})
    assert out["tickers"] == ["KO"] and out["weights"] == [2]


def test_the_repairs_compose_on_a_question_without_quantities():
    """Drop what was invented; nothing to restore since no weights were stated."""
    q = "should I add PLTR to my portfolio"
    parsed = {"intent": "should_buy_stock", "tickers": ["VTI", "JNJ"],
              "weights": [0.6, 0.4], "new_stock": "PLTR"}
    out = Q._repair_dropped_holdings(q, Q._drop_invented_holdings(q, parsed))
    assert out["tickers"] == []


def test_a_stated_portfolio_is_restored_but_not_second_guessed():
    """With quantities present the guard stands down, so an invented holding
    survives here. That is the deliberate trade: wrongly deleting a real
    holding is worse than keeping a spurious one the user can see listed."""
    q = "rank my portfolio: IVV 30%, BTC-USD 2%"
    parsed = {"intent": "rank_portfolio", "tickers": ["VTI"], "weights": [1.0],
              "new_stock": None}
    out = Q._repair_dropped_holdings(q, Q._drop_invented_holdings(q, parsed))
    assert "IVV" in out["tickers"] and "BTC-USD" in out["tickers"]


@pytest.mark.parametrize("question", [
    "evaluate my holdings: 40 percent Nvidia, 60 percent Coca Cola",
    "I hold half Apple and half Microsoft, how risky is that",
    "rank my portfolio: IVV 70%, KO 30%",
    "split equally between IVV and KO",
])
def test_stated_quantities_protect_real_holdings(question):
    """Holdings named by company ("40 percent Nvidia") resolve to symbols that
    are absent from the question text, so they look invented. A stated quantity
    means a stated portfolio: never drop in that case."""
    out = invented(question, intent="rank_portfolio",
                   tickers=["NVDA", "KO"], weights=[0.4, 0.6])
    assert out["tickers"] == ["NVDA", "KO"]


def test_without_a_quantity_the_guard_still_fires():
    out = invented("should I add PLTR to my portfolio",
                   tickers=["VTI", "JNJ"], weights=[0.6, 0.4], new_stock="PLTR")
    assert out["tickers"] == []
