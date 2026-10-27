"""Fetching the news sources so that one of them failing is survivable.

Each source is called behind its own guard and returns a SourceResult rather
than raising. The caller then composes whatever came back and tells the user
which sources contributed, because a degraded answer that looks identical to a
complete one is worse than an error.

Policy, in one place:
  * timeout   every outbound call is bounded, connect and read separately
  * retry     only on transient signals; a 401 fails the same way twice
  * parallel  three independent calls cost max(latency), not sum(latency)
  * floor     if nothing came back, say so instead of asking a model to improvise
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional
import logging
import time

import requests

logger = logging.getLogger("financeai.sources")

# (connect, read). Connect is short: a source that will not answer the handshake
# is down. Read is longer because these APIs are genuinely slow sometimes.
DEFAULT_TIMEOUT = (3.05, 8.0)

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 2            # one retry; these are read-only, best-effort reads
BACKOFF_SECONDS = 0.5


class SourceError(Exception):
    """A source refused. `status` lets the retry policy tell 429 from 401."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.status in RETRYABLE_STATUS


@dataclass
class SourceResult:
    name: str
    items: List[Any] = field(default_factory=list)
    ok: bool = True
    error: Optional[str] = None
    latency_ms: int = 0
    attempts: int = 1

    @property
    def usable(self) -> bool:
        return self.ok and bool(self.items)


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, SourceError):
        return exc.retryable
    # A socket that timed out or never connected is worth one more go.
    return isinstance(exc, (requests.Timeout, requests.ConnectionError))


def fetch_one(name: str, fn: Callable[[], List[Any]]) -> SourceResult:
    """Call one source, absorbing every failure into the result."""
    started = time.monotonic()
    last: Optional[Exception] = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            items = fn() or []
            return SourceResult(name, list(items), True, None,
                                int((time.monotonic() - started) * 1000), attempt)
        except Exception as e:                      # noqa: BLE001 - deliberate
            last = e
            if attempt < MAX_ATTEMPTS and _is_transient(e):
                logger.warning("%s failed (%s), retrying", name, e)
                time.sleep(BACKOFF_SECONDS)
                continue
            break

    logger.warning("%s unavailable: %s: %s", name, type(last).__name__, last)
    return SourceResult(name, [], False, f"{type(last).__name__}: {last}",
                        int((time.monotonic() - started) * 1000), attempt)


def gather(tasks: List[tuple]) -> List[SourceResult]:
    """Run `[(name, callable), ...]` concurrently. Order of results is preserved.

    Sequential calls cost the sum of their timeouts, which makes a generous
    timeout unaffordable. Concurrent, the worst case is one timeout.
    """
    if not tasks:
        return []
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        return list(pool.map(lambda t: fetch_one(t[0], t[1]), tasks))


def provenance(results: List[SourceResult]) -> str:
    """One line telling the user what the answer is actually based on."""
    used = [r.name for r in results if r.usable]
    empty = [r.name for r in results if r.ok and not r.items]
    failed = [r.name for r in results if not r.ok]

    parts = [f"Sources used: {', '.join(used) if used else 'none'}"]
    if empty:
        parts.append(f"returned nothing: {', '.join(empty)}")
    if failed:
        parts.append(f"unavailable: {', '.join(failed)}")
    return " | ".join(parts)
