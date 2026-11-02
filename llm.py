"""LLM provider abstraction with structured outputs and retry logic.

Providers
---------
* ``anthropic`` (default) - Anthropic Claude via the official SDK,
  ``client.messages.parse(..., output_format=<PydanticModel>)``.
* ``openai`` - OpenAI via the official SDK, ``client.responses.parse``.
* ``offline`` - no API calls; agents use the rule-based reasoner.

If a key is missing the client degrades to ``offline`` and records why, so
the prototype always runs. API keys are only read from environment variables.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

import config

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM cannot return a valid structured output."""


class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        requested = (provider or config.LLM_PROVIDER or "offline").lower()
        self.requested_provider = requested
        self.provider = "offline"
        self.model = ""
        self.note = ""
        self._client = None
        # token accounting (used to estimate cost in the evaluation dashboard)
        self.input_tokens = 0
        self.output_tokens = 0
        self.successful_calls = 0
        # Set after a non-retryable account/configuration error (billing, key, model): later calls fail fast
        # instead of repeating the same failing request for every review.
        self.fatal_error = ""

        if requested == "anthropic":
            if not config.ANTHROPIC_API_KEY:
                self.note = "ANTHROPIC_API_KEY not set - using offline rule-based reasoner."
            else:
                try:
                    import anthropic

                    self._client = anthropic.Anthropic(
                        api_key=config.ANTHROPIC_API_KEY, max_retries=2, timeout=config.LLM_TIMEOUT
                    )
                    self.provider, self.model = "anthropic", model or config.ANTHROPIC_MODEL
                except ImportError:
                    self.note = "anthropic package not installed - using offline reasoner."
        elif requested == "openai":
            if not config.OPENAI_API_KEY:
                self.note = "OPENAI_API_KEY not set - using offline rule-based reasoner."
            elif not (model or config.OPENAI_MODEL):
                self.note = "OPENAI_MODEL not set - using offline rule-based reasoner."
            else:
                try:
                    import openai

                    self._client = openai.OpenAI(api_key=config.OPENAI_API_KEY, max_retries=2, timeout=config.LLM_TIMEOUT)
                    self.provider, self.model = "openai", model or config.OPENAI_MODEL
                except ImportError:
                    self.note = "openai package not installed - using offline reasoner."
        elif requested != "offline":
            self.note = f"Unknown LLM_PROVIDER '{requested}' - using offline reasoner."

    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        return self.provider != "offline"

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}" if self.available else "offline-rule-based"

    # ------------------------------------------------------------------ #
    def structured(self, system: str, user: str, schema: Type[T]) -> T:
        """Return a validated instance of `schema`, retrying transient errors
        with exponential backoff. Raises LLMError on final failure."""
        if not self.available:
            raise LLMError("No LLM provider available (offline mode)")
        if self.fatal_error:
            raise LLMError(f"LLM disabled for this run: {self.fatal_error}")
        last_err: Exception | None = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                if self.provider == "anthropic":
                    return self._anthropic(system, user, schema)
                return self._openai(system, user, schema)
            except _NonRetryable as e:
                if "refusal" not in str(e):
                    self.fatal_error = explain_llm_error(str(e))
                    raise LLMError(self.fatal_error) from e
                raise LLMError(str(e)) from e
            except Exception as e:  # transient: rate limit, 5xx, timeout, invalid JSON
                last_err = e
                delay = config.LLM_RETRY_BASE_DELAY * (2 ** attempt)
                log.warning("LLM call failed (attempt %s/%s): %s - retrying in %.1fs",
                            attempt + 1, config.LLM_MAX_RETRIES, e, delay)
                if attempt + 1 < config.LLM_MAX_RETRIES:
                    time.sleep(delay)
        raise LLMError(f"LLM failed after {config.LLM_MAX_RETRIES} attempts: {last_err}")

    # ------------------------------------------------------------------ #
    def _anthropic(self, system: str, user: str, schema: Type[T]) -> T:
        import anthropic

        try:
            resp = self._client.messages.parse(
                model=self.model,
                max_tokens=config.LLM_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.NotFoundError, anthropic.BadRequestError) as e:
            raise _NonRetryable(f"{type(e).__name__}: {e}") from e
        if resp.stop_reason == "refusal":
            raise _NonRetryable("The model declined this request (stop_reason=refusal)")
        if resp.stop_reason == "max_tokens":
            raise LLMError("Output truncated (max_tokens) - increase LLM_MAX_TOKENS")
        self._record_usage(getattr(resp, "usage", None))
        parsed = resp.parsed_output
        if parsed is None:
            raise LLMError("No parsed output returned")
        return parsed

    def _record_usage(self, usage) -> None:
        if usage is None:
            return
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        self.successful_calls += 1

    def usage_snapshot(self) -> tuple[int, int, int]:
        return self.input_tokens, self.output_tokens, self.successful_calls

    def price_per_mtok(self) -> tuple[Optional[float], Optional[float]]:
        """Configured (input, output) USD price per million tokens for the active provider, or (None, None)."""
        if self.provider == "anthropic":
            return config.ANTHROPIC_PRICE_INPUT, config.ANTHROPIC_PRICE_OUTPUT
        if self.provider == "openai":
            return config.OPENAI_PRICE_INPUT, config.OPENAI_PRICE_OUTPUT
        if self.provider == "stub":
            return getattr(self, "stub_prices", (None, None))
        return None, None

    def _openai(self, system: str, user: str, schema: Type[T]) -> T:
        import openai

        try:
            resp = self._client.responses.parse(
                model=self.model, instructions=system, input=user, text_format=schema,
                max_output_tokens=config.LLM_MAX_TOKENS,
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError,
                openai.NotFoundError, openai.BadRequestError) as e:
            raise _NonRetryable(f"{type(e).__name__}: {e}") from e
        self._record_usage(getattr(resp, "usage", None))
        parsed = resp.output_parsed
        if parsed is None:
            raise LLMError("No parsed output returned")
        return parsed


class _NonRetryable(Exception):
    pass


def explain_llm_error(message: str) -> str:
    """Turn a provider error into a plain explanation (never includes the API key)."""
    m = message.lower()
    if "credit balance" in m or "billing" in m or "insufficient_quota" in m or "quota" in m:
        return ("The provider account has no remaining credit or billing is not set up (HTTP 400/429). Add credit in the "
                "provider console (Anthropic: console.anthropic.com → Plans & Billing); until then the rule-based fallback is used.")
    if "authentication" in m or "invalid x-api-key" in m or "401" in m:
        return "The API key was rejected (HTTP 401). Check ANTHROPIC_API_KEY in .env."
    if "permission" in m or "403" in m:
        return "The API key has no permission for this model (HTTP 403)."
    if "notfound" in m or "not_found" in m or "404" in m:
        return "The configured model was not found (HTTP 404). Check ANTHROPIC_MODEL in .env."
    if "badrequest" in m or "400" in m:
        return "The provider rejected the request (HTTP 400): " + message[:160]
    return message[:200]


class _Ping(BaseModel):
    ok: bool


def health_check(client: "LLMClient") -> dict:
    """One tiny structured call to verify key, model, billing and structured output. Costs a few tokens."""
    if not client.available:
        return {"ok": False, "message": client.note or "Offline mode: no LLM configured."}
    saved, client.fatal_error = client.fatal_error, ""
    try:
        client.structured("Reply with ok=true.", "Health check.", _Ping)
        return {"ok": True, "message": f"{client.name} answered a structured test request."}
    except LLMError as e:
        return {"ok": False, "message": str(e)}
    finally:
        client.fatal_error = client.fatal_error or saved


class StubLLM(LLMClient):
    """Test double that returns pre-programmed structured outputs (no network).
    Used in tests to exercise the LLM code path of every agent."""

    def __init__(self, responses: dict[str, list[BaseModel]] | None = None,
                 usage_per_call: tuple[int, int] = (0, 0), prices: tuple = (None, None)):
        super().__init__(provider="offline")
        self.provider, self.model = "stub", "stub-model"
        self.responses = responses or {}
        self.calls: list[tuple[str, str]] = []
        self.usage_per_call = usage_per_call
        self.stub_prices = prices

    def structured(self, system: str, user: str, schema: Type[T]) -> T:
        self.calls.append((schema.__name__, user[:80]))
        self.input_tokens += self.usage_per_call[0]
        self.output_tokens += self.usage_per_call[1]
        self.successful_calls += 1
        queue = self.responses.get(schema.__name__, [])
        if not queue:
            raise LLMError(f"StubLLM has no response queued for {schema.__name__}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        try:
            return schema.model_validate(item.model_dump() if isinstance(item, BaseModel) else item)
        except ValidationError as e:  # pragma: no cover
            raise LLMError(str(e)) from e
