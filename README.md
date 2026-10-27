<div align="center">

# Fina

**A reliability-first finance assistant built as two agents.**

Ask plain-English questions about a stock portfolio. One agent computes the
answer; a second reviews it against who you are. Every number comes from numpy
and scipy — never from a language model.

[![CI](https://github.com/KhabbabZakaria/FinanceAIAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/KhabbabZakaria/FinanceAIAgent/actions/workflows/ci.yml)
![tests](https://img.shields.io/badge/tests-167%20offline-brightgreen)
![eval](https://img.shields.io/badge/intent%20eval-100%25%20hosted%20%7C%2096.4%25%20local-brightgreen)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)

</div>

---

```
Q: should I buy NVDA? I hold IVV 60%, KO 40%

=== Should You Buy NVDA? ===                     ← AGENT A: deterministic
Your portfolio today, at your weights: 1.48
Best achievable Sharpe without NVDA:   1.58
Best achievable Sharpe with NVDA:      1.58
Marginal gain from adding NVDA:        +0.005
Decision: BUY

Suggested New Weights:
Ticker  Weight
   IVV  38.77%
    KO  58.10%
  NVDA   3.14%

--- Worth knowing ---                            ← AGENT B: critic
In the suggested new allocation, KO is 58%, above your 35% single-position
limit. The Sharpe improvement is +0.005, well inside the error of a one-year
estimate; this is close to a coin flip.
```

Agent A says BUY, because a Sharpe ratio went up. Agent B points out that the
improvement is statistical noise and the allocation breaks a constraint you set.
Neither agent is wrong; a user reading only the first one would be misled.

---

## Contents

[How it works](#how-it-works) · [The two agents](#the-two-agents) ·
[Reliability layer](#the-reliability-layer) · [Quick start](#quick-start) ·
[Configuration](#configuration) · [Tests](#tests) · [Evals](#evals) ·
[Design decisions](#design-decisions) · [Limitations](#limitations)

---

## How it works

```
  "should I buy NVDA? I hold IVV 60%, KO 40%"
          │
          ▼
  ┌──────────────────────────────────────────────┐
  │ QuestionClassifier    small LLM, JSON mode    │   LLM #1
  │ → intent + tickers + weights + topic          │
  └───────────────────────┬──────────────────────┘
                          ▼
  ┌──────────────────────────────────────────────┐
  │ Router          deterministic if/elif         │   no LLM
  │ dispatches only on known intents; one the     │
  │ model invents falls through, never executes   │
  └──┬─────────────────────────────────┬─────────┘
     ▼                                 ▼
  ┌───────────────────────────┐   ┌───────────────────────────────┐
  │ AGENT A    tools + maths  │   │ get_finance_news              │
  │ Yahoo Finance             │   │  NewsAPI ┐                    │
  │ numpy / scipy:            │   │  Reddit  ├ concurrent, each   │
  │   log returns, annualised │   │  Finnhub ┘ behind its own     │
  │   covariance, SLSQP       │   │            guard + timeout    │
  │   Sharpe optimisation     │   │          ▼                    │
  │                           │   │  grounded synthesis    LLM #2 │
  │ → answer + .facts         │   │  + "Sources used: ..."        │
  └────────────┬──────────────┘   └───────────────────────────────┘
               ▼
  ┌──────────────────────────────────────────────┐
  │ AGENT B    critic                             │
  │  checks()    pure Python rules vs profile.py  │   no LLM
  │              → exact, testable findings       │
  │  critique()  which matter, and why     LLM #3 │
  │  never recomputes; degrades to the rules      │
  └───────────────────────┬──────────────────────┘
                          ▼
             answer + "Worth knowing"
```

Three LLM calls, and none of them produce a number. Six intents —
`rank_portfolio`, `should_buy_stock`, `top_performers`, `stock_trend`,
`get_finance_news`, and `other`, which returns a capability list rather than a
blank page.

**The LLM never touches the maths.** Sharpe ratios, covariance matrices and
portfolio optimisation are numpy and scipy. The models turn a sentence into a
typed intent, summarise news they are handed, and explain findings they are
handed. That boundary is the main design decision in the project.

### Project layout

| File | Lines | What it is |
|---|---:|---|
| [`app.py`](app.py) | 348 | Flask routes, the intent router, watchlist seeding |
| [`helpers.py`](helpers.py) | 904 | Classifier, the four data sources, all portfolio maths |
| [`critic.py`](critic.py) | 249 | **Agent B** — deterministic rules + the LLM pass over them |
| [`sources.py`](sources.py) | 112 | Per-source isolation, timeouts, retry, provenance |
| [`llm.py`](llm.py) | 157 | Provider chain with failover |
| [`positions.py`](positions.py) | 177 | Lots you bought, and figures derived from them |
| [`conversations.py`](conversations.py) | 138 | Durable turn log; retrieval keyed on tickers |
| [`profile.py`](profile.py) | 38 | **Who you are** — the critic's only input about you |
| [`config.py`](config.py) | 52 | Environment, with defaults |
| [`monitoring_service.py`](monitoring_service.py) | 278 | Background price / volume / news alerts |
| [`tests/`](tests/) | 167 tests | Offline, no keys, a few seconds |
| [`evals/`](evals/) | 60 cases | Labelled set + scorer + regression gate |

---

## The two agents

### Agent A — compute the answer

Pulls live data and does the arithmetic. No model involved: log returns,
annualised covariance, SLSQP Sharpe optimisation, risk-free rate from `^TNX`.

It returns an `Analysis` — a `str` subclass carrying a `.facts` dict alongside
the formatted text. That small detail is what makes the second agent possible:
the critic reads exact figures instead of re-parsing prose or recomputing
anything.

```python
answer.facts
# {'kind': 'should_buy_stock', 'decision': 'BUY', 'marginal_gain': 0.0047,
#  'current_sharpe': 1.476, 'best_without': 1.579, 'best_with': 1.583,
#  'suggested_weights': {'IVV': 0.3877, 'KO': 0.5810, 'NVDA': 0.0314}, ...}
```

### Agent B — review it against the user

A second pass that reads those facts plus [`profile.py`](profile.py) and never
recomputes. The split *inside* it matters as much as the split between agents:

```
checks()     pure Python. Every constraint the profile can express is
             evaluated here, so findings are exact, testable, identical
             every run.

critique()   the model. Handed findings it did not have to derive, it
             decides which matter for this question and writes them in
             context.
```

Asked to critique a portfolio unaided, a 3B model produces generic hedging —
*"consider diversification"*. Handed concrete, pre-verified violations, it does
the one thing it is good at: prioritising and explaining.

Seven rules ship today: single-position cap, minimum position count,
speculative-bucket cap, cluster cap, fixed-income shortfall, and a flag for a
BUY whose Sharpe gain is inside the noise floor. **If no model answers, the raw
findings are rendered instead** — the critic degrades to a rule engine rather
than vanishing. If `checks()` itself throws, the answer goes out unreviewed. A
critic must never take the answer down with it.

---

## The reliability layer

**Provider fallback** — [`llm.py`](llm.py). Every LLM call walks a configurable
chain and falls through on failure. All tiers speak the OpenAI
chat-completions API, so one client type covers a local model and a hosted one;
only `base_url`, key and model differ. The result records which tiers were
skipped and why.

```bash
LLM_PROVIDERS=openai,ollama      # hosted first, local as the free fallback
```

**Graceful degradation across sources** — [`sources.py`](sources.py). The three
news APIs are fetched concurrently, each behind its own guard, returning a result
object rather than raising. One being down costs that source's contribution and
nothing else, and the answer says what it rests on:

```
[Sources used: Reddit, Finnhub | unavailable: NewsAPI]
```

Retries are transient-only — a 429 or 503 earns a second attempt, a 401 will
fail identically and is not retried. Every outbound call is bounded (3s connect,
8s read); concurrency is what makes a timeout that generous affordable, since
sequential calls would cost their sum. **If every source fails the request is
refused**, rather than handing an empty context to a model and letting it answer
from memory.

**Grounded synthesis** — the news prompt forbids facts not present in the
retrieved context. Asked for a P/E ratio that is not there, it says so instead of
inventing one. The wording was tuned against a small model, which over-refuses
under a stricter prompt.

**Weights are matched by name, never by position.** `yfinance` returns `Close`
columns alphabetically rather than in request order. Applying a weight vector
positionally silently values the wrong holdings — easy to do, invisible in the
output, and the single worst bug this project has had.

**Memory, for the critic only** — [`conversations.py`](conversations.py). Every
turn is written to a durable table, so on a server that stays up the record
accumulates indefinitely. Retrieval is keyed on **tickers, not similarity**:
asking about `BTC-USD` pulls every turn that named it, however old, plus the
last couple of turns for context. That is a `SELECT` — exact, cheap and
testable, where embeddings would make a precise lookup fuzzy for no gain.

History is evaluated as **rules**, not passed as prose. That distinction was
measured: handing the model 152 days of context produced near-identical output
with and without it, because every finding came from current weights. Turned
into a finding, both models use it:

```
[high] BTC-USD has been flagged 2 times before, first 152 days ago,
       and is still outside your limits.
```

**Rate limiting** — per client IP, `10 per minute;200 per day`. The per-minute
cap stops bursts; the daily cap is what actually protects the free-tier quotas,
which reset daily.

**Background monitoring** — price drops every 15 minutes against a rolling
24-hour baseline, volume spikes, and per-ticker breaking news. Jobs run once at
startup, so a misconfiguration surfaces immediately rather than an hour later.

---

## Quick start

Requires **Python 3.11+**.

### 1. Install

```bash
git clone https://github.com/KhabbabZakaria/FinanceAIAgent.git
cd FinanceAIAgent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run a model locally (free, offline, no key)

```bash
brew install ollama          # or https://ollama.com
ollama serve                 # leave running
ollama pull qwen2.5:3b       # ~1.9 GB
```

Needs roughly 2 GB of RAM while loaded. With no Ollama running it falls back to
OpenAI; with neither, questions return a clear error rather than a stack trace.

### 3. Add your API keys

```bash
cp .env.example .env
```

All four data sources have free tiers:

| Key | Where | Needed for |
|---|---|---|
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org) | news questions |
| `FINNHUB_API_KEY` | [finnhub.io](https://finnhub.io) | news + breaking-news alerts |
| `REDDIT_CLIENT_ID` / `_SECRET` | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) | sentiment |
| `OPENAI_API_KEY` | optional | fallback if Ollama is down |

Yahoo Finance needs no key, so **portfolio questions work with no keys at all.**

### 4. Tell it who you are  ← don't skip this

Edit [`profile.py`](profile.py). It is the only thing Agent B knows about you,
and **the committed file is a placeholder** — until you change it, the critic is
reasoning about a fictional person.

```python
USER_PROFILE = {
    "summary": "Long-horizon retail investor, investing monthly from salary...",
    "horizon_years": 20,
    "risk_tolerance": "high",
    "max_single_position": 0.35,   # ← rules use these numbers
    "max_speculative": 0.15,
    ...
}
```

Note the split: `summary`, `horizon_years` and `risk_tolerance` only reach the
**model**, shaping how findings are worded. The numeric fields drive the
**rules**, deciding what gets flagged at all. Writing *"I'm conservative"* in the
summary changes nothing on its own — you have to move the numbers too.

> `profile.py` is committed to git. If you make this repo public, keep real
> financial details out of it.

### 5. Set your starting watchlist

Edit `DEFAULT_WATCHLIST` at the top of [`app.py`](app.py). It seeds an empty
database only, so anything you later add or remove at `/watchlist` survives a
restart.

### 6. Run

```bash
python app.py        # http://localhost:5000
```

Ask it: *"should I buy NVDA? I hold IVV 60%, KO 40%"*

<details>
<summary><b>Optional: email alerts</b></summary>

The background monitor emails price-drop, volume and breaking-news alerts. It
needs `ALERT_EMAIL` and `ALERT_EMAIL_PASSWORD` in `.env`. For Gmail that must be
an **App Password** (myaccount.google.com → App passwords), not your account
password — a normal password is rejected at login. Failures are logged rather
than swallowed, so check `app-error.log` if nothing arrives.

Set `ENABLE_MONITORING=0` to run the web app without the scheduler.
</details>

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDERS` | `openai,ollama` | Fallback chain, in order |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Local model |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Local endpoint |
| `OPENAI_MODEL` | `gpt-4o-mini` | Hosted backup |
| `GROQ_API_KEY` / `_MODEL` | unset | Optional third tier; add `groq` to the chain |
| `ENABLE_CRITIC` | `1` | Agent B (one extra LLM call, ~8s) |
| `ENABLE_MONITORING` | `1` | Background scheduler |
| `QUESTION_RATE_LIMIT` | `10 per minute;200 per day` | Per-IP limit |
| `NEWS_ALERT_WINDOW_HOURS` | `1` | Breaking-news lookback |
| `DB_PATH` | `portfolios.db` | SQLite location |

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

```
167 passed in 1.73s
```

Fully offline — `yfinance`, HTTP and the provider chain are all stubbed — so CI
runs with no secrets and the result is identical every time. GitHub Actions runs
it on Python 3.11, 3.12 and 3.13 on every push.

| File | Tests | Covers |
|---|---:|---|
| `test_classifier.py` | 37 | Normalisation, JSON extraction, slot-repair rules |
| `test_sources.py` | 24 | Retry matrix, isolation, concurrency, provenance |
| `test_critic.py` | 23 | Each rule, boundaries, history rules, degradation |
| `test_positions.py` | 14 | Cost basis, holding period, benchmark excess, drawdown |
| `test_eval_harness.py` | 13 | The scorer, and that the golden set hasn't rotted |
| `test_routing.py` | 11 | Intent dispatch, blank-page fallback, news degradation |
| `test_monitoring.py` | 9 | Price seeding, ETF prices, alert thresholds |
| `test_conversations.py` | 9 | Durable log, ticker-keyed retrieval, no false matches |
| `test_llm_chain.py` | 7 | Provider failover, all-tiers-down, JSON mode |
| `test_rate_limit.py` | 6 | Per-IP limits, GET exemption, headers |
| `test_portfolio_math.py` | 6 | Weight alignment, missing tickers, BUY comparison |
| `test_watchlist_seed.py` | 5 | Seeds once, never clobbers user edits |
| `test_stock_trend.py` | 3 | Bounded walk-back, MultiIndex flattening |

Every file also passes on its own, not just as part of the suite — the fixtures
reset shared state rather than relying on import order.

Most are regressions for defects that actually shipped, which is why they read
oddly specifically:

| Test | The bug it pins down |
|---|---|
| `test_weights_follow_their_own_tickers_not_column_order` | `yfinance` returns columns alphabetically; weights were applied positionally, so the published Sharpe described a portfolio the user did not hold |
| `test_buy_verdict_compares_like_with_like` | BUY/DON'T BUY compared the user's weights against an *optimised* portfolio, crediting the new stock for gains that came from re-weighting. It could never return DON'T BUY |
| `test_walk_back_terminates_when_window_has_no_earlier_data` | An unbounded walk back to the previous trading day looped forever over a long market closure and hung the worker |
| `test_first_pass_records_a_baseline` | `price_history` was written only *after* an early `continue`, so the table was never seeded and no comparison was ever possible |
| `test_etf_price_is_read_from_regular_market_price` | ETFs and crypto leave `currentPrice` empty, so every non-stock holding was silently skipped |
| `test_company_news_is_called_with_a_symbol` | A wrong signature raised `TypeError` into a bare `except: pass`; breaking-news alerts had never fired once |
| `test_a_dropped_hyphenated_ticker_is_restored` | The model silently dropped `BTC-USD` from a 7-holding portfolio; normalisation then rescaled the survivors to 100%, so a 30% holding was reported as 30.61% and the loss was invisible |
| `test_all_sources_down_refuses_instead_of_improvising` | An empty context reaching the model means it answers from memory |
| `test_a_dead_model_degrades_to_the_rule_engine` | The critic must lose words, not the answer, when no provider responds |
| `test_classifier_failure_is_handled_not_a_500` | Classification sat outside the error handler, so any provider hiccup was an unhandled 500 |

---

## Evals

Unit tests stub the model out. They prove the code *around* it is correct and
say nothing about whether the model does its job. The eval harness is the other
half: a labelled set run through the real classifier, scored, compared against a
stored baseline.

```bash
python -m evals.run_intent                    # current provider chain
python -m evals.run_intent --provider openai  # compare a hosted model
python -m evals.run_intent --update-baseline  # accept the current scores
```

`qwen2.5:3b`, running locally:

```
intent accuracy    54/56  =  96.4%
slot accuracy      54/56  =  96.4%
hard failures        0
latency            p50 3.8s   p90 5.6s   max 6.9s

per intent:  should_buy_stock 100%  stock_trend 100%  top_performers 100%
             get_finance_news 100%  rank_portfolio 89%  other 90%
per tag:     easy 100%   paraphrase 100%   no-portfolio 100%
             contaminated-ticker 100%   compound 100%
             company-name 86%   adversarial 90%   malformed 75%
```

60 cases across six intents, deliberately weighted away from easy ones: company
names instead of tickers, weights that don't sum to 100, gibberish, a
prompt-injection attempt. A set everything passes measures nothing, so a unit
test fails if easy cases ever exceed half the dataset. Four cases tagged
`known-gap` document behaviour the product doesn't support (ranking by momentum,
comparing two stocks); they're reported separately and excluded from the score.

**The scores are fully reproducible.** Run three times end to end, all three
produced byte-identical accuracy and every individual failure recurred 3/3.
Nothing is sampling noise, which is what makes the 5pp regression gate
meaningful.

### What measurement found

**The model invents intents.** Asked for an account balance it returns
`account_balance`; asked to compare two stocks, `compare_stocks`. Neither exists
in the schema. The router dispatches only on known intents, so both fall through
to the capability list — containment working as designed, and now proven rather
than assumed.

**Company-name → ticker resolution is 73% accurate**, measured over 22 companies
× 3 reps:

```
Palantir               -> PALI    0/3      Snowflake              -> None   0/3
Palo Alto Networks     -> PNW     0/3      Advanced Micro Devices -> None   0/3
Salesforce             -> SFRI    0/3      Berkshire Hathaway     -> BRK.A  0/3
```

The split that matters is loud versus silent:

```
PALI   REAL  'Palisade Bio, Inc.'         $1.23     ← silent wrong answer
PNW    REAL  'Pinnacle West Capital'      $105.34   ← silent wrong answer
SFRI   not resolvable -> "No data found"            (loud, safe)
BRK.A  not resolvable -> "No data found"            (loud, safe)
```

**9% of company-name queries silently return real prices for the wrong
company.** Ask about Palantir, get a $1.23 biotech — no error, no warning. This
is the most dangerous failure in the system and **it is currently unguarded**;
see [Limitations](#limitations).

None of this is visible without measurement. That is the argument for the
harness.

---

## Design decisions

**A deterministic router, not an agent loop.** The model classifies; Python
decides. An agent free to choose its own tool calls would be more impressive to
describe and worse to rely on — where the output is a number someone might act
on, a wrong tool call is indistinguishable from a right one until the money
moves. Less autonomy, in exchange for failure modes that can be enumerated.

**Two agents, asymmetric.** Peers debating would double the latency and, over
deterministic arithmetic, have nothing to debate — an LLM can only talk itself
out of a correct number. A generator and a critic have genuinely different jobs,
and only one of them is allowed near the figures.

**Hosted by default, local as the fallback — because the eval said so.** The
same 60 cases were run against both models:

| | `qwen2.5:3b` (local) | `gpt-4o-mini` |
|---|---:|---:|
| intent accuracy | 96.4% | **100.0%** |
| slot accuracy | 96.4% | **98.2%** |
| latency p50 | 3.8s | **1.1s** |
| company name → ticker (22 firms) | 16/22 | **22/22** |
| cost | free | pennies/month |

The last row is the one that decided it. The 3B returns `PALI` for Palantir and
`PNW` for Palo Alto Networks — real tickers for the wrong companies, so the app
fetches genuine prices and presents them confidently. `gpt-4o-mini` made no such
error in 44 attempts.

I had assumed the small model was close enough on extraction, since that is the
task small models are supposed to be fine at. It is not: a 3.6-point intent gap
and a 27% ticker error rate, for 3.5x the latency. The local model remains a
good *fallback* — free, offline, and still 96.4% — which is why it stays second
in the chain rather than being removed.

Generation runs at ~14.8 tok/s on an M3, so on the local tier latency is roughly
`output_tokens / 14.8` — classification ~4s, critique ~8s, news synthesis ~20s.

**Rules in Python, judgement in the model.** Anything expressible as a rule is a
rule, so it is exact and testable. The model is reserved for what rules cannot
express: which of several true findings actually matters here. This decided
three separate designs — ticker lookup, position figures and conversation
history — and the last was measured rather than assumed: the same history as
prose context changed nothing; as a finding it changed the answer.

**Single-tenant.** The schema is user-scoped throughout
(`PRIMARY KEY (user_id, ticker)`, every query filtered), so multi-tenancy is a
deployment concern rather than a rewrite. Authentication was scoped out
deliberately to spend the time on the reliability layer.

**SQLite.** One writer, a few hundred rows a day. Postgres would be ceremony.

---

## Limitations

Known and measured, rather than discovered later.

- **Ticker hallucination is unguarded, and largely a small-model problem.** On
  `qwen2.5:3b`, 27% of company-name lookups are wrong and 9% silently return
  the wrong company's real prices (`Palantir` → `PALI`, a $1.23 biotech). On
  `gpt-4o-mini` the same 22 companies resolve 22/22. So running hosted avoids
  it in practice, but nothing in the code *prevents* it — a resolver over the
  SEC company list, with the model extracting the name span instead of
  recalling the symbol, is still the proper fix and is not built.
- **It is not a conversational chatbot.** Turns are logged and the critic reads
  them, but the **classifier is stateless** — it sees only the current question.
  "I hold AAPL 50%, MSFT 50%, should I buy NVDA?" works; the follow-up "what
  about TSLA?" does not, and is answered as a standalone question about TSLA
  with no memory that you hold anything. Pronouns, "that one" and "the second"
  are not resolved. Memory here means *"you have asked about this before and
  nothing changed"*, not dialogue.
- **Position history is stored but not yet collected.** [`positions.py`](positions.py)
  holds the schema and derives cost basis, holding period, return against the
  S&P and drawdown from entry, with 14 tests. Nothing populates it: there is no
  UI for entering lots and no `review_position` intent, so "I bought this five
  months ago, should I sell?" cannot be answered yet.
- **Momentum ranking is not implemented.** `top_performers` ranks a hardcoded
  cap list by one-year total return and ignores user-supplied tickers. Confirmed
  failing 3/3 in the eval and tagged `known-gap`.
- **Rate-limit and provider state are in-memory**, so they reset on restart and
  are per-process. Multiple workers would each keep their own count.
- **`price_history` grows without bound** — roughly 670 rows/day at seven
  tickers, never pruned.
- **Behind a reverse proxy every client looks the same** to the rate limiter
  until `ProxyFix` is added.
- **Flask's development server.** Fine for one user; a real deployment wants
  gunicorn, and the monitor would then need to run in exactly one process.

---

## Not investment advice

A Sharpe ratio computed from one year of daily closes is a backward-looking
statistic, not a forecast. The critic enforces constraints you wrote down; it
has no view on whether they are wise. This is a demonstration of engineering
practice, not a recommendation to trade.
