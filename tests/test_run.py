import json
import tempfile
import unittest
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

import run
from cola.client import OpenAIChatClient
from cola.data import StanceSample


class APITimeoutError(Exception):
    pass


class _FlakyClient:
    def __init__(self):
        self.calls = 0

    def complete(self, *, system, user):
        self.calls += 1
        if self.calls == 1:
            raise APITimeoutError("temporary network failure")
        return "A"


class _FlakyCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **request):
        self.calls += 1
        if self.calls == 1:
            raise APITimeoutError("temporary request failure")
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="A"),
                finish_reason="stop",
            )],
            model="test-model",
            usage=None,
        )


class RunTests(unittest.TestCase):
    def test_request_retry_records_attempt_count(self):
        client = OpenAIChatClient.__new__(OpenAIChatClient)
        client.model = "test-model"
        client._is_dashscope = False
        client.reasoning_effort = None
        client.max_completion_tokens = None
        client.max_retries = 1
        client.retry_backoff_seconds = 0
        client.calls = []
        client._lock = Lock()
        completions = _FlakyCompletions()
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        self.assertEqual(client.complete(system="system", user="user"), "A")
        self.assertEqual(client.calls[0]["attempts"], 2)

    def test_transient_sample_failure_is_retried(self):
        args = SimpleNamespace(sample_retries=1, retry_backoff=0, trace=False)
        result = run._run_one(
            StanceSample("sample-1", "text", "target", "against"),
            "direct",
            1,
            args,
            _FlakyClient(),
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["attempts"], 2)

    def test_keyboard_interrupt_writes_interrupted_record(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.jsonl"
            args = SimpleNamespace(
                repeats=1,
                limit=1,
                workers=1,
                sample_retries=0,
                request_retries=0,
                retry_backoff=0,
                csv=Path("data/smoke.csv"),
                output=output,
                method="direct",
                mock=True,
                model=None,
                env_file=None,
                trace=False,
            )
            with patch.object(run, "as_completed", side_effect=KeyboardInterrupt):
                self.assertEqual(
                    run.run(
                        csv=args.csv,
                        method=args.method,
                        limit=args.limit,
                        repeats=args.repeats,
                        workers=args.workers,
                        sample_retries=args.sample_retries,
                        request_retries=args.request_retries,
                        retry_backoff=args.retry_backoff,
                        mock=args.mock,
                        model=args.model,
                        env_file=args.env_file,
                        output=args.output,
                        trace=args.trace,
                    ),
                    130,
                )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows[-1]["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
