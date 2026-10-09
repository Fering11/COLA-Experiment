from __future__ import annotations

import random
import time
from pathlib import Path
from threading import Lock
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProviderConfig:
    api_key: str
    model: str
    base_url: str
    timeout_seconds: float = 60.0
    max_retries: int = 0
    max_completion_tokens: int | None = None
    reasoning_effort: str | None = None
    retry_backoff_seconds: float = 1.0


def load_provider_config(env_file: str | Path) -> ProviderConfig:
    """Read provider settings from an explicit env file, never from process env."""
    path = Path(env_file)
    if not path.is_file():
        raise FileNotFoundError(f"Provider config file does not exist: {path}")

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key.replace("_", "").isalnum():
            raise ValueError(f"Invalid provider config at {path}:{line_number}")
        values[key] = _strip_quotes(value.strip())

    def required(key: str) -> str:
        value = values.get(key, "").strip()
        if not value:
            raise ValueError(f"Missing {key} in provider config: {path}")
        return value

    return ProviderConfig(
        api_key=required("OPENAI_API_KEY"),
        model=required("OPENAI_MODEL"),
        base_url=required("OPENAI_BASE_URL"),
        timeout_seconds=_float_value(values, "OPENAI_TIMEOUT_SECONDS", 60.0),
        max_retries=_nonnegative_int(values, "OPENAI_MAX_RETRIES", 0),
        max_completion_tokens=_optional_positive_int(values, "OPENAI_MAX_COMPLETION_TOKENS"),
        reasoning_effort=values.get("OPENAI_REASONING_EFFORT") or None,
        retry_backoff_seconds=_float_value(values, "OPENAI_RETRY_BACKOFF_SECONDS", 1.0),
    )


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _float_value(values: dict[str, str], key: str, default: float) -> float:
    value = values.get(key, "").strip()
    parsed = default if not value else float(value)
    if parsed < 0:
        raise ValueError(f"{key} must be non-negative")
    return parsed


def _nonnegative_int(values: dict[str, str], key: str, default: int) -> int:
    value = values.get(key, "").strip()
    parsed = default if not value else int(value)
    if parsed < 0:
        raise ValueError(f"{key} must be non-negative")
    return parsed


def _optional_positive_int(values: dict[str, str], key: str) -> int | None:
    value = values.get(key, "").strip()
    if not value:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{key} must be positive")
    return parsed


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str) -> str:
        """Return one text completion for a system/user message pair."""


class OpenAIChatClient:
    """Small adapter around the OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        env_file: str | Path,
    ) -> None:
        config = load_provider_config(env_file)

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI client is not installed. Run: "
                "python -m pip install -r requirements.txt"
            ) from exc

        self.config = config
        self.model = config.model
        self._is_dashscope = "dashscope" in config.base_url or "maas.aliyuncs.com" in config.base_url
        self.max_retries = config.max_retries
        self.retry_backoff_seconds = config.retry_backoff_seconds
        self.calls: list[dict] = []
        self._lock = Lock()
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            # Retries are performed here so each attempt is recorded exactly once.
            max_retries=0,
        )
        self.settings = {
            "model": self.model,
            "base_url": config.base_url,
            "temperature": 0,
            "timeout_seconds": config.timeout_seconds,
            "max_retries": self.max_retries,
            "sdk_max_retries": 0,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "reasoning_effort": config.reasoning_effort,
            "max_completion_tokens": config.max_completion_tokens,
        }

    def complete(self, *, system: str, user: str) -> str:
        request = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.config.reasoning_effort:
            request["reasoning_effort"] = self.config.reasoning_effort
        if self.config.max_completion_tokens is not None:
            token_key = "max_tokens" if self._is_dashscope else "max_completion_tokens"
            request[token_key] = self.config.max_completion_tokens
        started = time.perf_counter()
        record = {"system": system, "user": user, "attempts": 0}
        try:
            for attempt in range(self.max_retries + 1):
                record["attempts"] = attempt + 1
                try:
                    response = self._client.chat.completions.create(**request)
                    choice = response.choices[0]
                    content = choice.message.content
                    if choice.finish_reason == "length":
                        raise RuntimeError("Completion truncated; increase or remove the token cap.")
                    if not content:
                        raise RuntimeError("The model returned an empty completion.")
                    record.update(
                        status="ok", content=content, finish_reason=choice.finish_reason,
                        returned_model=response.model,
                        usage=response.usage.model_dump() if response.usage else None,
                    )
                    return content.strip()
                except Exception as exc:
                    record.update(
                        status="error", error_type=type(exc).__name__,
                        http_status=getattr(exc, "status_code", None),
                    )
                    if attempt >= self.max_retries or not is_retryable_error(exc):
                        raise
                    delay = min(self.retry_backoff_seconds * (2 ** attempt), 30.0)
                    time.sleep(random.uniform(0, delay))
        finally:
            record["elapsed_s"] = round(time.perf_counter() - started, 4)
            with self._lock:
                self.calls.append(record)


def is_retryable_error(exc: BaseException) -> bool:
    """Return whether an exception is likely to be fixed by retrying the request."""
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 425, 429} or (
        isinstance(status, int) and status >= 500
    ):
        return True
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectionError",
        "ReadTimeout",
        "TimeoutError",
    }


class DeterministicMockClient:
    """Offline fixture that exercises orchestration without claiming ML quality."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        system_lower = system.lower()
        user_lower = user.lower()

        if "final stance judge" in system_lower or "stance classifier" in system_lower:
            if (
                "only way i support hillary" in user_lower
                or "no more republicans" in user_lower
            ):
                return "A"
            if (
                "explaining feminism" in user_lower
                or "major setback for @epa" in user_lower
            ):
                return "B"
            return "C"

        if "linguist" in system_lower:
            return "Mock linguistic analysis: grammar, mood, and lexical choices."
        if "domain specialist" in system_lower:
            return "Mock domain analysis: entities, events, and target relations."
        if "social media" in system_lower:
            return "Mock social analysis: hashtags, tone, slang, and implication."
        if "stance debater" in system_lower:
            return "Mock debate: three cited clues support the assigned stance."
        raise AssertionError(f"Unexpected system prompt: {system}")
