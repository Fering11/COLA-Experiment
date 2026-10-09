from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from cola.client import DeterministicMockClient, OpenAIChatClient, is_retryable_error
from cola.data import load_csv
from cola.metrics import classification_metrics
from cola.pipeline import COLAPipeline, parse_judge_option


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "sem16_train.csv"
DEFAULT_ENV_FILE = ROOT / ".hy3.env"


@dataclass(frozen=True)
class RunConfig:
    method: str
    csv: Path
    limit: int | None
    repeats: int
    workers: int
    sample_retries: int
    retry_backoff: float
    mock: bool
    env_file: Path | None
    output: Path
    trace: bool


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auditable COLA reproduction runner")
    p.add_argument("--method", choices=("cola", "direct", "both"), default="cola")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--limit", type=int)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--workers", type=int, default=2, help="Maximum concurrent pipelines")
    p.add_argument("--sample-retries", type=int, default=1, help="Retries for transient failures of a complete sample")
    p.add_argument("--retry-backoff", type=float, default=1.0, help="Maximum base delay for request retries")
    p.add_argument("--mock", action="store_true", help="Use deterministic offline fixture")
    p.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Provider env file (defaults to local .dp.env when present)",
    )
    p.add_argument("--output", type=Path, default=ROOT / "results" / "run.jsonl")
    p.add_argument("--trace", action="store_true", help="Include prompts and model responses")
    return p


def write(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


class ProgressReporter:
    """Render progress from the writer thread so worker output cannot interleave."""

    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = total
        self.completed = 0
        self.successes = 0
        self.failures = 0

    def update(self, record: dict) -> None:
        self.completed += 1
        if record["status"] == "ok":
            self.successes += 1
        else:
            self.failures += 1
        print(
            f"\r{self.label}: {self.completed}/{self.total} "
            f"ok={self.successes} errors={self.failures}",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def finish(self, interrupted: bool = False) -> None:
        suffix = " interrupted" if interrupted else " complete"
        print(f"{suffix}", file=sys.stderr, flush=True)


def make_client(config: RunConfig):
    if config.mock:
        return DeterministicMockClient()
    return OpenAIChatClient(env_file=config.env_file)


def run_direct(sample, client):
    raw = client.complete(
        system="You are a stance classifier. Return only one option: A: Against, B: Favor, or C: Neutral.",
        user=f"Target: {sample.target}\nSentence: {sample.text}\nReturn the most accurate option only.",
    )
    return parse_judge_option(raw), {"raw": raw}


def _run_one(sample, method: str, repeat: int, config: RunConfig, client) -> dict:
    item = {
        "status": "ok",
        "method": method,
        "repeat": repeat,
        "id": sample.sample_id,
        "gold": sample.label,
        "attempts": 0,
    }
    for attempt in range(config.sample_retries + 1):
        item["attempts"] = attempt + 1
        try:
            if method == "cola":
                result = COLAPipeline(client).predict(
                    text=sample.text,
                    target=sample.target,
                    id=str(sample.sample_id),
                )
                prediction = result.label
                if config.trace:
                    item["trace"] = result.to_dict()
            else:
                prediction, detail = run_direct(sample, client)
                if config.trace:
                    item["trace"] = detail
            item["status"] = "ok"
            item.pop("error_type", None)
            item.pop("retryable", None)
            item["prediction"] = prediction
            return item
        except Exception as exc:
            retryable = is_retryable_error(exc)
            item.update(
                status="error",
                error_type=type(exc).__name__,
                retryable=retryable,
            )
            if attempt >= config.sample_retries or not retryable:
                return item
            time.sleep(min(config.retry_backoff * (2 ** attempt), 30.0))
    return item


def _write_summary(handle, records: list[dict], method: str, repeat: int, started: float, client, config: RunConfig) -> None:
    successful = [record for record in records if record["status"] == "ok"]
    truth = [record["gold"] for record in successful]
    predictions = [record["prediction"] for record in successful]
    summary = {
        "status": "summary",
        "method": method,
        "repeat": repeat,
        "samples": len(records),
        "successes": len(successful),
        "failures": len(records) - len(successful),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "metrics": classification_metrics(truth, predictions) if predictions else None,
    }
    if not config.mock and isinstance(client, OpenAIChatClient):
        summary["provider"] = client.settings
    write(handle, summary)
    print(json.dumps(summary, ensure_ascii=False))


def run(
    *,
    csv: str | Path = DEFAULT_CSV,
    method: str = "cola",
    limit: int | None = None,
    repeats: int = 1,
    workers: int = 2,
    sample_retries: int = 1,
    retry_backoff: float = 1.0,
    mock: bool = False,
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    output: str | Path = ROOT / "results" / "run.jsonl",
    trace: bool = False,
) -> int:
    """Run independent samples concurrently with explicit, reusable parameters."""
    config = RunConfig(
        method=method,
        csv=Path(csv),
        limit=limit,
        repeats=repeats,
        workers=workers,
        sample_retries=sample_retries,
        retry_backoff=retry_backoff,
        mock=mock,
        env_file=Path(env_file) if env_file is not None else None,
        output=Path(output),
        trace=trace,
    )
    if config.method not in {"cola", "direct", "both"}:
        raise ValueError("method must be one of: cola, direct, both")
    if config.repeats < 1 or (config.limit is not None and config.limit < 1):
        raise ValueError("--repeats and --limit must be positive")
    if config.workers < 1:
        raise ValueError("--workers must be positive")
    if config.sample_retries < 0:
        raise ValueError("sample_retries must be non-negative")
    if config.retry_backoff < 0:
        raise ValueError("--retry-backoff must be non-negative")

    samples = load_csv(config.csv, limit=config.limit)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    methods = ("cola", "direct") if config.method == "both" else (config.method,)
    worker_count = min(config.workers, len(samples))

    with config.output.open("w", encoding="utf-8", newline="\n") as handle:
        for repeat in range(1, config.repeats + 1):
            for method in methods:
                client = make_client(config)
                print(f"Using model: {getattr(client, 'model', 'mock')}")
                records: list[dict] = []
                started = time.perf_counter()
                progress = ProgressReporter(
                    f"{method} repeat {repeat}", len(samples)
                )
                executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="cola")
                futures = []
                interrupted = False
                try:
                    futures = [
                        executor.submit(_run_one, sample, method, repeat, config, client)
                        for sample in samples
                    ]
                    for future in as_completed(futures):
                        record = future.result()
                        records.append(record)
                        write(handle, record)
                        progress.update(record)
                except KeyboardInterrupt:
                    interrupted = True
                    for future in futures:
                        future.cancel()
                    write(handle, {
                        "status": "interrupted",
                        "method": method,
                        "repeat": repeat,
                        "completed": len(records),
                        "pending": len(futures) - len(records),
                    })
                    handle.flush()
                    progress.finish(interrupted=True)
                    return 130
                else:
                    handle.flush()
                    progress.finish()
                    _write_summary(handle, records, method, repeat, started, client, config)
                    handle.flush()
                finally:
                    executor.shutdown(wait=not interrupted, cancel_futures=interrupted)
    return 0


def run_model_suite() -> None:
    """Run the four explicitly configured provider files while preserving labels."""
    sem16_dataset = ROOT / "data" / "sem16_train.csv"
    run(csv=sem16_dataset, method="cola", trace=True, output=ROOT / "results" / "sem16" / "dpv4.jsonl", env_file=ROOT / ".dp.env", limit=1)
    run(csv=sem16_dataset, method="cola", trace=True, output=ROOT / "results" / "sem16" / "glm5.3.jsonl", env_file=ROOT / ".glm5.3.env", limit=1)
    run(csv=sem16_dataset, method="cola", trace=True, output=ROOT / "results" / "sem16" / "hy4.jsonl", env_file=ROOT / ".hy4.env", limit=1)
    run(csv=sem16_dataset, method="cola", trace=True, output=ROOT / "results" / "sem16" / "hy3.jsonl", env_file=ROOT / ".hy3.env", limit=1)

run_model_suite()

if __name__ == "__main__" and 0:
    try:
        cli = parser().parse_args()
        raise SystemExit(
            run(
                csv=cli.csv,
                method=cli.method,
                limit=cli.limit,
                repeats=cli.repeats,
                workers=cli.workers,
                sample_retries=cli.sample_retries,
                retry_backoff=cli.retry_backoff,
                mock=cli.mock,
                env_file=cli.env_file,
                output=cli.output,
                trace=cli.trace,
            )
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
