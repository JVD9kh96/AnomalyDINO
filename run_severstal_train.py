"""Train frozen-DINOv2 linear patch classifier on Severstal folds."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from src.evaluation.cross_validation import load_config
from src.evaluation.reproducibility import (
    clear_cuda_memory,
    save_folds_json,
    save_json,
    seed_all,
)
from src.severstal.dataset import SeverstalDataset
from src.training.data import LazySampleList
from src.training.sampling import resolve_training_ids
from src.training.trainer import PatchClassifierTrainer, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Severstal linear patch classifier (frozen DINOv2)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/severstal_dino_linear_probe.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Train a single fold only (0-indexed). Default: all folds.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override data.root from config.",
    )
    return parser.parse_args()


def run_training(
    config: dict,
    fold_indices: list[int] | None = None,
    config_path: str | Path | None = None,
) -> dict:
    seed = config.get("seed", 42)
    data_cfg = config["data"]
    cv_cfg = config["cv"]
    patch_cfg = config.get("patch_eval", {})
    detector_cfg = config["detector"]
    train_cfg = config.get("train", {})

    if detector_cfg.get("name", "dino_linear_probe") != "dino_linear_probe":
        raise ValueError(
            "run_severstal_train.py expects detector.name=dino_linear_probe, "
            f"got {detector_cfg.get('name')!r}."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(train_cfg.get("output_dir", "results_train")) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    if config_path:
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f)

    folds_json_path = run_dir / "folds.json"
    dataset = SeverstalDataset(
        data_root=data_cfg["root"],
        image_shape=tuple(data_cfg.get("image_shape", [256, 1600])),
        num_classes=data_cfg.get("num_classes", 4),
        n_folds=cv_cfg.get("n_folds", 5),
        seed=seed,
        stratify=cv_cfg.get("stratify", True),
        shuffle=cv_cfg.get("shuffle", True),
        folds_json_path=folds_json_path if folds_json_path.exists() else None,
    )
    if not folds_json_path.exists():
        save_folds_json(dataset.fold_splits, folds_json_path)

    n_folds = cv_cfg.get("n_folds", 5)
    if fold_indices is None:
        fold_indices = list(range(n_folds))

    all_results: dict = {}
    for fold_idx in fold_indices:
        fold_seed = seed + fold_idx
        seed_all(fold_seed)
        print(f"\n=== Fold {fold_idx} (seed={fold_seed}) ===")

        fold_dir = run_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_ids, val_ids = dataset.get_fold_split(fold_idx)
        shots = detector_cfg.get("shots", 8)
        ref_sampling = detector_cfg.get("reference_sampling", "class_balanced")
        fit_ids = resolve_training_ids(
            dataset,
            fold_idx,
            shots,
            fold_seed,
            reference_sampling=ref_sampling,
        )

        if shots is None:
            print("  Mode: full supervised (all train-fold images)")
        elif shots == -1:
            print(
                f"  Mode: all eligible train images "
                f"(shots=-1, sampling={ref_sampling})"
            )
        else:
            print(f"  Mode: k-shot supervised training (shots={shots})")
        print(
            f"  Fit images: {len(fit_ids)}, "
            f"Train fold: {len(train_ids)}, Val: {len(val_ids)}"
        )

        train_samples = LazySampleList(dataset, fit_ids)
        val_samples = LazySampleList(dataset, val_ids)

        tcfg = TrainConfig.from_dicts(
            detector_cfg,
            train_cfg=train_cfg,
            patch_eval=patch_cfg,
            data_cfg=data_cfg,
            seed=fold_seed,
        )
        trainer = PatchClassifierTrainer(tcfg)

        checkpoint_path = detector_cfg.get("checkpoint_path")
        if checkpoint_path:
            print(f"  Loading checkpoint: {checkpoint_path}")
            trainer.load_checkpoint(checkpoint_path)
            result = {
                "optimal_threshold": trainer.optimal_threshold,
                "loaded_checkpoint": str(checkpoint_path),
            }
        else:
            result = trainer.fit(
                train_samples=train_samples,
                val_samples=val_samples,
                output_dir=fold_dir,
            )

        result["fold"] = fold_idx
        result["n_fit"] = len(fit_ids)
        result["n_train_fold"] = len(train_ids)
        result["n_val"] = len(val_ids)
        save_json(result, fold_dir / "train_summary.json")
        all_results[f"fold_{fold_idx}"] = result
        print(
            f"  Done. optimal_threshold={result.get('optimal_threshold')}"
        )

        # Release backbone / classifier before the next fold
        trainer.cleanup()
        del trainer, train_samples, val_samples
        clear_cuda_memory()

    save_json(all_results, run_dir / "summary.json")
    return all_results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root:
        config.setdefault("data", {})["root"] = args.data_root

    fold_indices = [args.fold] if args.fold is not None else None

    print("Severstal Linear Probe Training")
    print(f"  Config: {args.config}")
    print(f"  Data root: {config['data']['root']}")
    print(f"  Folds: {fold_indices if fold_indices else 'all'}")
    print(f"  Mode: {config['detector'].get('classification_mode', 'binary')}")

    results = run_training(config, fold_indices=fold_indices, config_path=args.config)
    print("\n=== Training complete ===")
    for key, val in results.items():
        print(f"  {key}: threshold={val.get('optimal_threshold')}")


if __name__ == "__main__":
    main()
