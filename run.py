from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
import rich

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


def run(args) -> int:
    if args.repeats < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("--repeats and --limit must be positive")
    samples = load_csv(args.csv, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    methods = ("cola", "direct") if args.method == "both" else (args.method,)
    with args.output.open("w", encoding="utf-8") as handle:
        for repeat in range(1, args.repeats + 1):
            for method in methods:
                client = make_client(args)
                pipeline = COLAPipeline(client) if method == "cola" else None
                truth, predictions = [], []
                started = time.perf_counter()
                for sample in samples:
                    
                    rich.print(f"[bold blue]Processing sample {sample.sample_id}[/bold blue]")

                    item = {"status": "ok", "method": method, "repeat": repeat, "id": sample.sample_id, "gold": sample.label}
                    try:
                        if method == "cola":
                            result = pipeline.predict(text=sample.text, target=sample.target,printFunc=rich.print)
                            prediction = result.label
                            if args.trace:
                                item["trace"] = result.to_dict()
                        else:
                            prediction, detail = run_direct(sample, client)
                            if args.trace:
                                item["trace"] = detail
                        truth.append(sample.label)
                        predictions.append(prediction)
                        item["prediction"] = prediction
                    except Exception as exc:
                        item.update(status="error", error_type=type(exc).__name__)
                    write(handle, item)
                summary = {"status": "summary", "method": method, "repeat": repeat,
                           "samples": len(samples), "successes": len(predictions),
                           "failures": len(samples) - len(predictions),
                           "elapsed_s": round(time.perf_counter() - started, 3),
                           "metrics": classification_metrics(truth, predictions) if predictions else None}
                if not args.mock and isinstance(client, OpenAIChatClient):
                    summary["provider"] = client.settings
                write(handle, summary)
                print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(parser().parse_args()))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
