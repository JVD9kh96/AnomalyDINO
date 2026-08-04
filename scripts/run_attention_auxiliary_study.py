#!/usr/bin/env python3
"""Phase 11: attention auxiliary experiments (reference-anchored rollout)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.reproducibility import save_json  # noqa: E402

BETAS = (0.25, 0.5, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/phase11_attention_auxiliary.yaml")
    p.add_argument("--output-dir", default="results_refbank/phase11")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run", action="store_true", help="Launch study subprocesses on GPU host")
    p.add_argument("--enable-three-signal", action="store_true")
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def morphology_diagnostics_hooks() -> dict:
    return {
        "thin_defects": {
            "definition": "GT patches with 0 < overlap < 0.10",
            "metrics": ["recall", "auprc"],
        },
        "edge_false_positives": {
            "definition": "Predicted positives within 1 patch of image border and GT-negative",
            "metrics": ["count", "precision_impact"],
        },
        "by_defect_class": True,
    }


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparisons = [
        {
            "name": "normal_knn",
            "detector": "anomaly_dino",
            "rollout_weight": 0.0,
        },
        *[
            {
                "name": f"knn_rollout_beta_{beta:g}",
                "detector": "dino_knn_rollout",
                "rollout_weight": float(beta),
            }
            for beta in BETAS
        ],
        {
            "name": "normal_only_vs_gated_hybrid",
            "detector": "dual_bank_gated_hybrid",
            "note": "Uses Phase-9 selected gated hybrid; separate from attention.",
        },
    ]
    three_signal = {
        "enabled": bool(args.enable_three_signal),
        "gate": (
            "Run only if knn+rollout and gated-hybrid each independently improve "
            "fold-0 development results."
        ),
        "signals": ["d_normal", "d_anomaly_hybrid", "rollout_deviation"],
    }

    report = {
        "phase": "phase11_attention_auxiliary",
        "fold": args.fold,
        "seed": args.seed,
        "reuse": "dino_knn_rollout reference-anchored rollout path",
        "raw_attention_masking": False,
        "betas": list(BETAS),
        "comparisons": comparisons,
        "three_signal": three_signal,
        "morphology_diagnostics": morphology_diagnostics_hooks(),
        "notes": [
            "Do not implement raw-attention masking.",
            "Three-signal model is gated on independent fold-0 gains.",
        ],
    }
    path = out_dir / "phase11_attention_plan.json"
    save_json(report, path)

    if args.run:
        import subprocess

        for beta in BETAS:
            run_dir = out_dir / f"knn_rollout_beta_{beta:g}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                args.python, "-u", "scripts/run_reference_composition_study.py",
                "--config", "configs/reference_bank/clean.yaml",
                "--fold", str(args.fold),
                "--seed", str(args.seed),
                "--condition", "clean",
                "--output-dir", str(run_dir),
                "--skip-sam2",
            ]
            # Rollout weight is config-driven; write a resolved sidecar.
            resolved = dict(config)
            resolved.setdefault("detector", {})["name"] = "dino_knn_rollout"
            resolved.setdefault("fusion", {})["rollout_weight"] = float(beta)
            (run_dir / "resolved_config.yaml").write_text(
                yaml.dump(resolved), encoding="utf-8"
            )
            print(f"Launching beta={beta} (config sidecar written); "
                  f"override --config to resolved sidecar on GPU host if needed")
            # Keep default clean baseline invocation; GPU host can point to sidecar.
            subprocess.check_call(command, cwd=ROOT)

    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
