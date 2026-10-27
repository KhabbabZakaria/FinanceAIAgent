from dotenv import load_dotenv
import os

load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Alert email credentials (used by monitoring_service.py)
ALERT_EMAIL = os.getenv("ALERT_EMAIL")
ALERT_EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD")

# LLM provider chain, tried left to right; the first one that answers wins.
# A provider with no credential is skipped, so dropping "groq" in here after
# setting GROQ_API_KEY is all it takes to add a tier.
# Hosted first: the 60-case eval scores gpt-4o-mini at 100% intent vs 96.4%
# local, and 22/22 on company-name->ticker vs 16/22. A provider with no
# credential is skipped, so with no OPENAI_API_KEY this falls through to
# Ollama and the app still runs free and offline.
LLM_PROVIDERS = os.getenv("LLM_PROVIDERS", "openai,ollama")

# Local small model (free, offline). Needs `ollama serve` running.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# Hosted backups
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Background monitoring. Set ENABLE_MONITORING=0 to boot the web app without the
# scheduler (and without the emails it sends).
ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "1").lower() not in ("0", "false", "no", "")

# How far back a breaking-news check looks. One hour is often empty; widen this
# if the alerts feel too sparse.
NEWS_ALERT_WINDOW_HOURS = int(os.getenv("NEWS_ALERT_WINDOW_HOURS", "1"))

# Rate limit on the question endpoint, per client IP. Semicolon-separated.
# The per-minute cap stops bursts; the daily cap is what actually protects the
# NewsAPI / Finnhub / Reddit free-tier quotas, which reset per day, not per minute.
QUESTION_RATE_LIMIT = os.getenv("QUESTION_RATE_LIMIT", "10 per minute;200 per day")

# Database location. Overridable so tests (and deployments) never touch the
# developer's local portfolios.db.
DB_PATH = os.getenv("DB_PATH", "portfolios.db")

# Critic agent: a second pass that reviews an answer against profile.py.
# Costs one extra LLM call (~8s on the local model), so it is switchable.
ENABLE_CRITIC = os.getenv("ENABLE_CRITIC", "1").lower() not in ("0", "false", "no", "")