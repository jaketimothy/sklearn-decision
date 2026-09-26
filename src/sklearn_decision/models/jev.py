"""TypeSafe's Jev over its HTTP API (``/v1/systemone``)."""
from __future__ import annotations

import asyncio
import os
import random
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from .._answers import DecisionModelError, DistAnswer, NoulAnswer, Response, opt_float
from .._async import run_coro
from .._cache import canon
from .base import Capabilities, DecisionModel

__all__ = ["JevModel", "JevAPIError", "PRICE_PER_INPUT_MTOK", "DEFAULT_BASE_URL"]

PRICE_PER_INPUT_MTOK = 0.042  # USD, Sept 2026 list price; output tokens are free
DEFAULT_BASE_URL = "https://api.typesafe.ai"
MAX_CHOICE_OPTIONS = 255  # API limit today; raise it here when TypeSafe does
MAX_SCORE_LEVELS = 10

_sleep = asyncio.sleep  # indirection so tests can skip backoff waits


class JevAPIError(DecisionModelError):
    def __init__(self, status: int, body: str):
        super().__init__(f"Jev API error {status}: {body[:500]}", fatal=status in (401, 403))
        self.status = status
        self.body = body


class JevModel(DecisionModel):
    """TypeSafe Jev as a decision model.

    Parameters
    ----------
    name : str
        Model version. Pin a concrete one (default "jev-1.13"). "jev-latest"
        works, but the cache only sees the requested name, so a silent
        upgrade would mix versions in one feature matrix.
    api_key : str or None
        Falls back to ``TYPESAFE_API_KEY``. Read when a request is made, so
        estimators can be fitted and cloned without credentials.
    base_url : str or None
        Falls back to ``TYPESAFE_BASE_URL``, then the TypeSafe API. Pointing
        it at "https://openrouter.ai/api" also works.
    timeout : float
        Per-request timeout in seconds.
    max_retries : int
        Retries for 408/409/429/5xx and transport errors, with exponential
        backoff that honours ``Retry-After``.
    max_concurrency : int
        Requests in flight at once.
    max_questions_per_call : int
        Questions per request. Each request re-sends the state, so fewer,
        larger requests are cheaper. The state plus all questions must fit
        the per-request token limit.
    transport : httpx.AsyncBaseTransport or None
        Custom transport, e.g. ``httpx.MockTransport`` in tests.
    max_cost_usd : float or None
        Spending cap for this model instance (one per fitted estimator).
        Before each batch, the estimated cost of the uncached requests is
        added to the spend so far (from reported input tokens); if the total
        would exceed the cap, nothing is sent and a fatal
        :class:`DecisionModelError` is raised.
    """

    def __init__(self, name: str = "jev-1.13", *, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 30.0, max_retries: int = 5, max_concurrency: int = 16,
                 max_questions_per_call: int = 50, transport: httpx.AsyncBaseTransport | None = None,
                 max_cost_usd: float | None = None):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrency = max_concurrency
        self.max_questions_per_call = max_questions_per_call
        self.transport = transport
        self.max_cost_usd = max_cost_usd

    # ---------------- DecisionModel API ----------------

    def resolve(self) -> JevModel:
        if not isinstance(self.name, str) or not self.name.startswith("jev-"):
            raise ValueError(f"name must be a Jev model name such as 'jev-1.13', got {self.name!r}")
        if self.name.endswith("latest"):
            warnings.warn(f"model {self.name!r} is not pinned: cached answers from different versions "
                          "can end up in one feature matrix. Pin a concrete version.", stacklevel=3)
        for p in ("max_retries",):
            if not isinstance(getattr(self, p), int) or getattr(self, p) < 0:
                raise ValueError(f"{p} must be an int >= 0")
        for p in ("max_concurrency", "max_questions_per_call"):
            if not isinstance(getattr(self, p), int) or getattr(self, p) < 1:
                raise ValueError(f"{p} must be an int >= 1")
        if not self.timeout or self.timeout <= 0:
            raise ValueError("timeout must be > 0")
        if self.max_cost_usd is not None and not self.max_cost_usd >= 0:
            raise ValueError("max_cost_usd must be >= 0 or None")
        return self

    def cache_namespace(self) -> str:
        return f"jev:{self.name}"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            max_choice_options=MAX_CHOICE_OPTIONS,
            max_score_levels=MAX_SCORE_LEVELS,
            max_questions_per_call=self.max_questions_per_call,
            native_confidence=True,
            deterministic=False,  # published tests put the noul noise floor around 0.01
        )

    def answer(self, items: Sequence[tuple[Any, Mapping[str, dict]]]) -> list[Response | BaseException]:
        if not items:
            return []
        if self.max_cost_usd is not None:
            spent = self.usage["input_tokens"] / 1e6 * PRICE_PER_INPUT_MTOK
            batch = sum(_est_tokens(s, q) for s, q in items) / 1e6 * PRICE_PER_INPUT_MTOK
            if spent + batch > self.max_cost_usd:
                raise DecisionModelError(
                    f"Spending cap: this batch (~${batch:.4f}) on top of ${spent:.4f} spent would exceed "
                    f"max_cost_usd={self.max_cost_usd}. Nothing was sent.", fatal=True)
        raw = run_coro(self._fetch_all(items))
        out: list[Response | BaseException] = []
        for (_, questions), res in zip(items, raw):
            if isinstance(res, BaseException):
                out.append(res)
                continue
            try:
                out.append(self._parse(questions, res))
            except Exception as e:  # malformed response for this item only
                out.append(DecisionModelError(f"Could not parse Jev response: {e!r}"))
        return out

    def estimate_cost(self, states, requests, chars_per_token: float = 4.0) -> dict:
        """Rough upper bound with an empty cache."""
        q_tok = [len(canon(r)) / chars_per_token for r in requests]
        s_tok = [_state_chars(s) / chars_per_token for s in states]
        total = sum(st + qt for st in s_tok for qt in q_tok)
        return {
            "rows": len(states),
            "requests": len(states) * len(requests),
            "est_input_tokens": int(total),
            "est_cost_usd": round(total / 1e6 * PRICE_PER_INPUT_MTOK, 4),
            "max_state_tokens": int(max(s_tok, default=0)),
        }

    # ---------------- internals ----------------

    def _parse(self, questions: Mapping[str, dict], res: Mapping) -> Response:
        usage = res.get("usage") or {}
        self.usage["calls"] += 1
        self.usage["input_tokens"] += int(usage.get("input_tokens", 0))
        self.usage["output_tokens"] += int(usage.get("output_tokens", 0))
        raw = res["answers"]
        answers = {}
        for q, spec in questions.items():
            if q not in raw:
                raise KeyError(f"question {q!r} missing from response")
            answers[q] = _normalize(spec, raw[q])
        self.usage["answers_fetched"] += len(answers)
        return Response(answers, str(res.get("model") or self.name))

    def _endpoint(self) -> str:
        base = self.base_url or os.environ.get("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL
        return base.rstrip("/") + "/v1/systemone"

    async def _fetch_all(self, items) -> list:
        key = self.api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise DecisionModelError("Set TYPESAFE_API_KEY or pass JevModel(api_key=...)", fatal=True)
        sem = asyncio.Semaphore(self.max_concurrency)
        headers = {"Authorization": f"Bearer {key}"}
        async with httpx.AsyncClient(headers=headers, timeout=self.timeout, transport=self.transport) as client:
            coros = [self._post(client, sem, state, dict(questions)) for state, questions in items]
            return await asyncio.gather(*coros, return_exceptions=True)

    async def _post(self, client: httpx.AsyncClient, sem: asyncio.Semaphore, state, questions) -> dict:
        payload = {"model": self.name, "state": state, "questions": questions}
        err: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            retry_after = None
            async with sem:
                try:
                    r = await client.post(self._endpoint(), json=payload)
                except httpx.TransportError as e:
                    r, err = None, e
            if r is not None:
                if r.status_code == 200:
                    return r.json()
                err = JevAPIError(r.status_code, r.text)
                if r.status_code not in (408, 409, 429) and r.status_code < 500:
                    raise err  # bad request / auth: retrying won't help
                try:
                    retry_after = float(r.headers.get("retry-after", ""))
                except ValueError:
                    retry_after = None
            if attempt == self.max_retries:
                break
            wait = retry_after if retry_after is not None else 0.5 * 2**attempt
            await _sleep(wait * (1 + 0.25 * random.random()))
        raise err  # type: ignore[misc]


def _state_chars(state) -> int:
    return len(state if isinstance(state, str) else canon(state))


def _est_tokens(state, questions, chars_per_token: float = 4.0) -> float:
    return (_state_chars(state) + len(canon(questions))) / chars_per_token


def _normalize(spec: Mapping, raw: Mapping):
    """Jev's wire answer -> the package's answer schema."""
    t = spec["type"]
    if t == "noul":
        return NoulAnswer(float(raw["noul"]))
    probs = raw.get("probabilities") or {}
    if t == "choice":
        labels = [str(k) for k in spec["criteria"]]
    else:  # score probabilities are keyed by level index
        labels = [str(i) for i in range(len(spec["criteria"]))]
    return DistAnswer(tuple(float(probs.get(lb, 0.0)) for lb in labels), opt_float(raw.get("confidence")))
