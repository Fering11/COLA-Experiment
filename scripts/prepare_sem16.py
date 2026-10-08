"""Convert SemEval-2016 Task 6 TSV files to COLA-Experiment CSV.

The converter keeps the original split and target text, normalizes labels,
and refuses malformed or unknown-label rows instead of silently dropping them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


LABELS = {
    "FAVOR": "favor",
    "AGAINST": "against",
    "NONE": "neutral",
    "NEUTRAL": "neutral",
    "favor": "favor",
    "against": "against",
    "neutral": "neutral",
}


def pick(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] is not None:
            return row[name].strip()
    raise ValueError(f"missing one of columns: {', '.join(names)}")


def convert(source: Path, destination: Path) -> dict[str, object]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        if not fields:
            raise ValueError("input has no header")
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for line_no, raw in enumerate(reader, start=2):
            try:
                sample_id = pick(raw, "ID", "id", "Tweet ID", "tweet_id")
                target = pick(raw, "Target", "target")
                text = pick(raw, "Tweet", "tweet", "text", "Text")
                original_label = pick(raw, "Stance", "stance", "label", "Label")
            except ValueError as exc:
                raise ValueError(f"line {line_no}: {exc}") from exc
            label = LABELS.get(original_label.strip())
            if label is None:
                raise ValueError(f"line {line_no}: unsupported stance {original_label!r}")
            if not sample_id or not target or not text:
                raise ValueError(f"line {line_no}: id, target, and text must be non-empty")
            if sample_id in seen:
                raise ValueError(f"line {line_no}: duplicate id {sample_id!r}")
            seen.add(sample_id)
            rows.append({"id": sample_id, "text": text, "target": target, "label": label})

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text", "target", "label"])
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {label: 0 for label in ("favor", "against", "neutral")}
    targets: dict[str, int] = {}
    for row in rows:
        counts[row["label"]] += 1
        targets[row["target"]] = targets.get(row["target"], 0) + 1
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {"source": str(source), "output": str(destination), "rows": len(rows),
            "labels": counts, "targets": targets, "sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare SemEval-2016 Task 6 data for COLA-Experiment")
    parser.add_argument("source", type=Path, help="Original SemEval TSV file")
    parser.add_argument("--output", type=Path, required=True, help="COLA CSV output path")
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
