from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.evaluation.reproducibility import to_json_serializable


def save_history_json(history: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_serializable(history), f, indent=2)


def save_history_csv(history: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return

    keys: list[str] = []
    seen: set[str] = set()
    for row in history:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {k: _csv_value(row.get(k)) for k in keys}
            )


def save_history(
    history: list[dict[str, Any]],
    output_dir: str | Path,
    log_format: str = "both",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_format = log_format.lower()
    if log_format in ("json", "both"):
        save_history_json(history, output_dir / "history.json")
    if log_format in ("csv", "both"):
        save_history_csv(history, output_dir / "history.csv")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return f"{value:.8g}"
    return value
