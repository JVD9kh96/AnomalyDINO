from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, KFold


def seed_all(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_fold_splits(
    image_ids: list[str],
    labels: list[int],
    n_folds: int,
    seed: int,
    stratify: bool = True,
    shuffle: bool = True,
) -> list[tuple[list[str], list[str]]]:
    """Return list of (train_ids, val_ids) per fold."""
    indices = np.arange(len(image_ids))
    if stratify:
        splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=shuffle, random_state=seed
        )
        splits = splitter.split(indices, labels)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
        splits = splitter.split(indices)

    fold_splits = []
    for train_idx, val_idx in splits:
        train_ids = [image_ids[i] for i in train_idx]
        val_ids = [image_ids[i] for i in val_idx]
        fold_splits.append((train_ids, val_ids))
    return fold_splits


def save_folds_json(
    fold_splits: list[tuple[list[str], list[str]]],
    output_path: str | Path,
) -> None:
    """Persist fold assignments as image_id -> fold index (validation fold)."""
    folds: dict[str, int] = {}
    for fold_idx, (_, val_ids) in enumerate(fold_splits):
        for image_id in val_ids:
            folds[image_id] = fold_idx

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2, sort_keys=True)


def load_folds_json(path: str | Path) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_json_serializable(obj):
    """Convert numpy types and nested structures to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_serializable(data), f, indent=2)
