from __future__ import annotations

import argparse
import json
import sys
import time
import rich
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cola.client import DeterministicMockClient, OpenAIChatClient
from cola.data import load_csv
from cola.metrics import classification_metrics
from cola.pipeline import COLAPipeline, parse_judge_option


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "smoke.csv"
DEFAULT_DEV = ROOT / "data" / "dev15.csv"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auditable COLA reproduction runner")
    p.add_argument("--method", choices=("cola", "direct", "both"), default="cola")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--limit", type=int)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--workers", type=int, default=4, help="Maximum concurrent pipelines")
    p.add_argument("--mock", action="store_true", help="Use deterministic offline fixture")
    p.add_argument("--model")
    p.add_argument("--env-file", type=Path, help="Provider env file, e.g. ../COLA-Research/qwen.env")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "run.jsonl")
    p.add_argument("--trace", action="store_true", help="Include prompts and model responses")
    return p


def write(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def make_client(args):
    return DeterministicMockClient() if args.mock else OpenAIChatClient(model=args.model, env_file=args.env_file)


def run_direct(sample, client):
    raw = client.complete(
        system="You are a stance classifier. Return only one option: A: Against, B: Favor, or C: Neutral.",
        user=f"Target: {sample.target}\nSentence: {sample.text}\nReturn the most accurate option only.",
    )
    return parse_judge_option(raw), {"raw": raw}


def _run_one(sample, method: str, repeat: int, args, client) -> dict:
    item = {
        "status": "ok",
        "method": method,
        "repeat": repeat,
        "id": sample.sample_id,
        "gold": sample.label,
    }
    try:
        if method == "cola":
            print(f"Processing {sample.sample_id}")
            result = COLAPipeline(client).predict(
                text=sample.text,
                target=sample.target,
                printFunc=rich.print,
                id=str(sample.sample_id)
            )
            prediction = result.label
            if args.trace:
                item["trace"] = result.to_dict()
        else:
            prediction, detail = run_direct(sample, client)
            if args.trace:
                item["trace"] = detail
        item["prediction"] = prediction
    except Exception as exc:
        item.update(status="error", error_type=type(exc).__name__)
    return item


def _write_summary(handle, records: list[dict], method: str, repeat: int, started: float, client, args) -> None:
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
    if not args.mock and isinstance(client, OpenAIChatClient):
        summary["provider"] = client.settings
    write(handle, summary)
    print(json.dumps(summary, ensure_ascii=False))


def run(args) -> int:
    """Run independent samples concurrently and write completed records in one thread."""
    if args.repeats < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("--repeats and --limit must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    samples = load_csv(args.csv, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    methods = ("cola", "direct") if args.method == "both" else (args.method,)
    workers = min(args.workers, len(samples))

    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for repeat in range(1, args.repeats + 1):
            for method in methods:
                client = make_client(args)
                records: list[dict] = []
                started = time.perf_counter()
                executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cola")
                futures = [
                    executor.submit(_run_one, sample, method, repeat, args, client)
                    for sample in samples
                ]
                try:
                    for future in as_completed(futures):
                        record = future.result()
                        records.append(record)
                        write(handle, record)
                except KeyboardInterrupt:
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
                    executor.shutdown(wait=False, cancel_futures=True)
                    return 130
                else:
                    executor.shutdown(wait=True, cancel_futures=False)
                    handle.flush()
                    _write_summary(handle, records, method, repeat, started, client, args)
                    handle.flush()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(parser().parse_args()))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
