from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Lock
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str) -> str:
        """Return one text completion for a system/user message pair."""


class OpenAIChatClient:
    """Small adapter around the OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        env_file: str | Path | None = None,
    ) -> None:
        _load_local_dotenv(env_file)
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Use --mock for an offline run."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI client is not installed. Run: "
                "python -m pip install -r requirements.txt"
            ) from exc

        self.model = model or os.getenv("OPENAI_MODEL", "qwen-plus")
        endpoint = base_url or os.getenv("OPENAI_BASE_URL") or None
        self._is_dashscope = bool(endpoint and ("dashscope" in endpoint or "maas.aliyuncs.com" in endpoint))
        timeout_value = timeout
        if timeout_value is None:
            timeout_value = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
        self.reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT") or None
        self.max_completion_tokens = _optional_int("OPENAI_MAX_COMPLETION_TOKENS")
        max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "0"))
        self.calls: list[dict] = []
        self._lock = Lock()
        self._client = OpenAI(
            api_key=key,
            base_url=endpoint,
            timeout=timeout_value,
            max_retries=max_retries,
        )
        self.settings = {
            "model": self.model,
            "base_url": str(self._client.base_url),
            "temperature": 0,
            "timeout_seconds": timeout_value,
            "max_retries": max_retries,
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_completion_tokens,
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
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        if self.max_completion_tokens is not None:
            token_key = "max_tokens" if self._is_dashscope else "max_completion_tokens"
            request[token_key] = self.max_completion_tokens
        started = time.perf_counter()
        record = {"system": system, "user": user}
        try:
            response = self._client.chat.completions.create(**request)
            choice = response.choices[0]
            content = choice.message.content
            record.update(
                status="ok", content=content, finish_reason=choice.finish_reason,
                returned_model=response.model,
                usage=response.usage.model_dump() if response.usage else None,
            )
            if choice.finish_reason == "length":
                raise RuntimeError("Completion truncated; increase or remove the token cap.")
            if not content:
                raise RuntimeError("The model returned an empty completion.")
            return content.strip()
        except Exception as exc:
            # API exception bodies can contain credentials; persist only safe metadata.
            record.update(status="error", error_type=type(exc).__name__,
                          http_status=getattr(exc, "status_code", None))
            raise
        finally:
            record["elapsed_s"] = round(time.perf_counter() - started, 4)
            with self._lock:
                self.calls.append(record)


def _optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _load_local_dotenv(env_file: str | Path | None = None) -> None:
    """Load an explicit or nearby env file without overriding exported variables."""
    if env_file:
        candidates = [Path(env_file)]
    elif os.getenv("COLA_ENV_FILE"):
        candidates = [Path(os.environ["COLA_ENV_FILE"])]
    else:
        candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        if env_file or os.getenv("COLA_ENV_FILE"):
            raise FileNotFoundError("The specified env file does not exist.")
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_simple_dotenv(path)
    else:
        load_dotenv(path, override=False)


def _load_simple_dotenv(path: Path) -> None:
    """Small fallback for the KEY=value files used by this project."""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator or not key.isidentifier() or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value


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
