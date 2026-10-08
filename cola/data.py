from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


LABEL_ALIASES = {
    "favor": "favor",
    "pro": "favor",
    "support": "favor",
    "against": "against",
    "con": "against",
    "oppose": "against",
    "neutral": "neutral",
    "none": "neutral",
}


@dataclass(frozen=True)
class StanceSample:
    sample_id: str
    text: str
    target: str
    label: str


def normalize_label(label: str) -> str:
    key = label.strip().lower()
    try:
        return LABEL_ALIASES[key]
    except KeyError as exc:
        accepted = ", ".join(sorted(LABEL_ALIASES))
        raise ValueError(f"Unknown label {label!r}; accepted labels: {accepted}") from exc


def load_csv(path: str | Path, *, limit: int | None = None) -> list[StanceSample]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    samples: list[StanceSample] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"text", "target", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row_index, row in enumerate(reader, start=1):
            if None in row or any(row.get(key) is None for key in required):
                raise ValueError(f"Malformed CSV at data row {row_index}")
            if any(not row[key].strip() for key in required):
                raise ValueError(f"Empty required field at data row {row_index}")
            sample_id = (row.get("id") or str(row_index)).strip()
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"Empty or duplicate sample id at data row {row_index}")
            seen_ids.add(sample_id)
            samples.append(
                StanceSample(
                    sample_id=sample_id,
                    text=row["text"].strip(),
                    target=row["target"].strip(),
                    label=normalize_label(row["label"]),
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise ValueError("CSV contains no samples")
    return samples
