#!/usr/bin/env python3
"""Run Phase 5 exact memory-budget and clean-shot controls on fold 0."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL_BUDGET = 51_200  # 8 clean references × 6,400 DINO patches at 448 px.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/phase5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--phase4-report",
        default="results/phase4/phase4_compact_purification_controls_report.json",
        help="Phase 4 report whose recommended purification setting is consumed by default",
    )
    parser.add_argument(
        "--phase2-report",
        default="results_reference_composition/phase2/phase2_purification_quality_report.json",
        help="Phase 2 report used to resolve the selected oracle overlap rule",
    )
    parser.add_argument(
        "--purification-mode",
        choices=("auto_purified", "fixed_ratio_trim"),
        default=None,
        help="Explicit override of the Phase 4-selected purification mode",
    )
    parser.add_argument("--purification-percentile", type=float, default=None)
    parser.add_argument("--fixed-trim-fraction", type=float, default=None)
    parser.add_argument(
        "--oracle-overlap-rule",
        choices=("any_overlap", "at_least_10_percent", "at_least_50_percent"),
        default=None,
        help="Oracle overlap rule for oracle_greedy_budget rows (default: Phase 2 selected)",
    )
    parser.add_argument(
        "--append-rows",
        action="store_true",
        help=(
            "Only run/aggregate the additive policy-matched rows and merge them into "
            "an existing Phase 5 report without rewriting prior rows."
        ),
    )
    parser.add_argument(
        "--include-optional-4plus8",
        action="store_true",
        help="Also include optional naive_greedy_budget_4plus8 control.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def specs(purification_mode: str) -> list[dict]:
    result = [
        {"name": f"clean_{shots}", "condition": "clean", "clean_shots": shots, "additional_shots": 0, "budget_policy": "greedy_coreset", "budget": FINAL_BUDGET, "filter": "none"}
        for shots in (1, 2, 4, 8)
    ]
    result.extend(
        [
            # Establish how much of a gain is simply the larger unbudgeted bank.
            {"name": "expanded_full_2plus8", "condition": "contaminated_all", "clean_shots": 2, "additional_shots": 8, "budget_policy": None, "budget": None, "filter": "none"},
            # Exact-size random subsampling control at the 8-clean patch budget.
            {"name": "expanded_random_budget_2plus8", "condition": "contaminated_all", "clean_shots": 2, "additional_shots": 8, "budget_policy": "random", "budget": FINAL_BUDGET, "filter": "none"},
            # Same deterministic greedy policy and budget for purified expansion.
            {"name": "purified_budget_1plus8", "condition": purification_mode, "clean_shots": 1, "additional_shots": 8, "budget_policy": "greedy_coreset", "budget": FINAL_BUDGET, "filter": "selected_purification"},
            {"name": "purified_budget_2plus8", "condition": purification_mode, "clean_shots": 2, "additional_shots": 8, "budget_policy": "greedy_coreset", "budget": FINAL_BUDGET, "filter": "selected_purification"},
            {"name": "purified_budget_4plus8", "condition": purification_mode, "clean_shots": 4, "additional_shots": 8, "budget_policy": "greedy_coreset", "budget": FINAL_BUDGET, "filter": "selected_purification"},
        ]
    )
    return result


def additive_specs(
    *,
    include_optional_4plus8: bool = True,
    oracle_condition: str = "oracle_purified",
) -> list[dict]:
    """Policy-matched exact-budget rows added without rerunning existing Phase-5 rows."""
    rows = [
        {
            "name": "naive_greedy_budget_2plus8",
            "condition": "contaminated_all",
            "clean_shots": 2,
            "additional_shots": 8,
            "budget_policy": "greedy_coreset",
            "budget": FINAL_BUDGET,
            "filter": "none",
        },
        {
            "name": "random20_greedy_budget_2plus8",
            "condition": "random_filtered",
            "clean_shots": 2,
            "additional_shots": 8,
            "budget_policy": "greedy_coreset",
            "budget": FINAL_BUDGET,
            "filter": "random_reject_20",
            "fixed_trim_fraction": 0.20,
        },
        {
            "name": "oracle_greedy_budget_2plus8",
            "condition": oracle_condition,
            "clean_shots": 2,
            "additional_shots": 8,
            "budget_policy": "greedy_coreset",
            "budget": FINAL_BUDGET,
            "filter": "oracle_overlap",
        },
    ]
    if include_optional_4plus8:
        rows.append(
            {
                "name": "naive_greedy_budget_4plus8",
                "condition": "contaminated_all",
                "clean_shots": 4,
                "additional_shots": 8,
                "budget_policy": "greedy_coreset",
                "budget": FINAL_BUDGET,
                "filter": "none",
            }
        )
    return rows


def active_specs(
    purification_mode: str,
    *,
    append_rows: bool,
    include_optional_4plus8: bool,
) -> list[dict]:
    if append_rows:
        return additive_specs(include_optional_4plus8=include_optional_4plus8)
    return [
        *specs(purification_mode),
        *additive_specs(include_optional_4plus8=include_optional_4plus8),
    ]

def _ensure_cuda(device: str | None) -> None:
    if device is not None and not device.startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run Phase 5 in the GPU screen session.")


def _run(command: list[str], label: str) -> None:
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT)
    while True:
        try:
            code = process.wait(timeout=60)
            break
        except subprocess.TimeoutExpired:
            print(f"  [{label}] still running ({(time.monotonic() - started) / 60:.1f} min)", flush=True)
    if code:
        raise subprocess.CalledProcessError(code, command)
    print(f"  [{label}] completed in {(time.monotonic() - started) / 60:.1f} min", flush=True)


def manifest_path(output_dir: Path, clean_shots: int, additional_shots: int) -> Path:
    return output_dir / f"phase5_f0_s42_clean{clean_shots}_additional{additional_shots}_manifest.json"


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def selected_purification(args: argparse.Namespace) -> dict:
    """Resolve an explicit override or consume Phase 4's reported recommendation."""
    if args.purification_mode is not None:
        setting = {"condition": args.purification_mode}
        if args.purification_mode == "auto_purified":
            if args.purification_percentile is None:
                raise ValueError("--purification-percentile is required for auto_purified")
            setting["percentile"] = args.purification_percentile
        else:
            if args.fixed_trim_fraction is None:
                raise ValueError("--fixed-trim-fraction is required for fixed_ratio_trim")
            setting["trim_fraction"] = args.fixed_trim_fraction
        return setting

    recommendation = _load(resolve(args.phase4_report)).get("recommended_setting", {})
    condition = recommendation.get("condition")
    if condition not in {"auto_purified", "fixed_ratio_trim"}:
        raise ValueError("Phase 4 recommendation is not a supported Phase 5 purification mode")
    setting = {"condition": condition, "source": "phase4_recommended_setting"}
    if condition == "auto_purified":
        setting["percentile"] = recommendation["percentile"]
    else:
        setting["trim_fraction"] = recommendation["trim_fraction"]
    return setting


def run(args: argparse.Namespace, output_dir: Path, purification: dict) -> None:
    _ensure_cuda(args.device)
    prepared_manifests: set[tuple[int, int]] = set()
    study_specs = active_specs(
        purification["condition"],
        append_rows=bool(args.append_rows),
        include_optional_4plus8=bool(args.include_optional_4plus8),
    )
    for index, spec in enumerate(study_specs, start=1):
        key = (spec["clean_shots"], spec["additional_shots"])
        manifest = manifest_path(output_dir, *key)
        if key not in prepared_manifests:
            if not manifest.is_file():
                command = [
                    args.python, "-u", "scripts/phase0_freeze_paired_inputs.py", "--config",
                    "configs/phase0_paired_reference_manifest.yaml", "--fold", "0", "--seed",
                    str(args.seed), "--clean-shots", str(key[0]), "--additional-shots", str(key[1]),
                    "--output", str(manifest),
                ]
                print(f"[manifest clean={key[0]} additional={key[1]}] freezing", flush=True)
                _run(command, "manifest")
            prepared_manifests.add(key)

        run_dir = output_dir / f"phase5_f0_{spec['name']}_s{args.seed}"
        if not args.rerun and (run_dir / "metrics.json").is_file():
            print(f"[run {index}/{len(study_specs)}: {spec['name']}] reusing completed output", flush=True)
            continue
        command = [
            args.python, "-u", "scripts/run_reference_composition_study.py", "--config",
            (
                "configs/reference_bank/proposed_distance20.yaml"
                if spec["condition"] == "fixed_ratio_trim"
                else "configs/reference_bank/auto_purified.yaml"
            ),
            "--fold", "0", "--seed", str(args.seed),
            "--split-seed", str(args.split_seed), "--condition", spec["condition"],
            "--clean-shots", str(spec["clean_shots"]), "--additional-shots", str(spec["additional_shots"]),
            "--output-dir", str(run_dir), "--paired-manifest", str(manifest), "--skip-sam2",
        ]
        if spec["condition"] == "auto_purified":
            command.extend(("--acceptance-percentile", str(purification["percentile"])))
        if spec["condition"] == "fixed_ratio_trim":
            command.extend(("--fixed-trim-fraction", str(purification["trim_fraction"])))
        if spec["condition"] == "random_filtered":
            # Match distance20 retention via fixed_trim_fraction consumed as random matched size.
            trim = float(spec.get("fixed_trim_fraction", purification.get("trim_fraction", 0.20)))
            command.extend(("--fixed-trim-fraction", str(trim)))
        if spec["budget"] is not None:
            command.extend(("--coreset-size", str(spec["budget"]), "--budget-policy", spec["budget_policy"]))
        if args.device is not None:
            command.extend(("--device", args.device))
        print(f"[run {index}/{len(study_specs)}: {spec['name']}] starting", flush=True)
        _run(command, spec["name"])


def _row_from_metrics(spec: dict, path: Path) -> dict:
    metrics = _load(path)
    if metrics.get("fold") != 0 or metrics.get("seed") != 42 or metrics.get("split_seed") != 42:
        raise ValueError(f"Invalid Phase 5 provenance: {path}")
    bank = metrics["reference"].get("bank_stats", {})
    return {
        **spec,
        "path": str(path),
        "auprc": metrics["patch"].get("auprc"),
        "auroc": metrics["patch"].get("auroc"),
        "fixed_f1": metrics["patch"].get("fixed_threshold", {}).get("f1"),
        "f1_max": metrics["patch"].get("f1_optimal", {}).get("f1"),
        "n_memory_patches_clean": bank.get("n_memory_patches_clean"),
        "n_candidate_patches_before_filter": bank.get("n_candidate_patches_before_filter"),
        "n_candidate_patches_after_filter": bank.get("n_candidate_patches_after_filter"),
        "n_memory_patches_before_budget": bank.get("n_memory_patches_before_budget"),
        "n_memory_patches_final": bank.get("n_memory_patches_final"),
        "budget_exact": (
            bank.get("n_memory_patches_final") == spec["budget"] if spec["budget"] else None
        ),
    }


def aggregate(
    output_dir: Path,
    purification: dict,
    *,
    append_rows: bool = False,
    include_optional_4plus8: bool = True,
) -> Path:
    report_path = output_dir / "phase5_exact_memory_budget_controls_report.json"
    study_specs = active_specs(
        purification["condition"],
        append_rows=append_rows,
        include_optional_4plus8=include_optional_4plus8,
    )
    new_rows = []
    for spec in study_specs:
        path = output_dir / f"phase5_f0_{spec['name']}_s42" / "metrics.json"
        if not path.is_file():
            if append_rows:
                print(f"Skipping missing additive row: {path}", flush=True)
                continue
            raise ValueError(f"Missing Phase 5 result: {path}")
        new_rows.append(_row_from_metrics(spec, path))

    if append_rows and report_path.is_file():
        existing = _load(report_path)
        prior_rows = list(existing.get("rows") or [])
        by_name = {row["name"]: row for row in prior_rows}
        for row in new_rows:
            by_name[row["name"]] = row
        # Preserve prior order, then append newly introduced names.
        ordered_names = [row["name"] for row in prior_rows]
        for row in new_rows:
            if row["name"] not in ordered_names:
                ordered_names.append(row["name"])
        rows = [by_name[name] for name in ordered_names]
        notes = list(existing.get("notes") or [])
    else:
        rows = new_rows
        notes = [
            "Clean 1/2/4-shot baselines are naturally below the 8-clean budget and are not upsampled.",
            "Expanded/purified comparisons at the target budget use exactly 51,200 final patches.",
            "F1-max is reported but should not be used to interpret budget-controlled gains alone.",
            "purified_budget_1plus8 may be under-budget (scarcity); do not upsample.",
            "Additive naive/random20/oracle rows share the 2+8 candidate IDs and greedy coreset.",
        ]

    additive_note = (
        "Appended additive policy-matched rows without rewriting prior Phase-5 evidence."
        if append_rows
        else "Full Phase-5 matrix including additive policy-matched rows."
    )
    if additive_note not in notes:
        notes.append(additive_note)

    report = {
        "phase": "phase5_exact_memory_budget_controls",
        "fold": 0,
        "seed": 42,
        "split_seed": 42,
        "sam2_skipped": True,
        "selected_purification": purification,
        "target_final_patch_budget": FINAL_BUDGET,
        "append_rows": bool(append_rows),
        "rows": rows,
        "notes": notes,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def resolve_oracle_rule(args: argparse.Namespace) -> str:
    if args.oracle_overlap_rule:
        return args.oracle_overlap_rule
    path = resolve(args.phase2_report)
    if path.is_file():
        report = _load(path)
        return str(report.get("selected_oracle_rule") or report.get("canonical_ground_truth_rule") or "any_overlap")
    return "any_overlap"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.seed != 42 or args.split_seed != 42:
        raise ValueError("Phase 5 is intentionally restricted to fold 0, seed 42, split seed 42")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    purification = selected_purification(args)
    oracle_rule = resolve_oracle_rule(args)
    purification = {**purification, "oracle_overlap_rule": oracle_rule}
    print(f"Phase 5 selected purification: {purification}", flush=True)
    print(f"Phase 5 append_rows={args.append_rows}", flush=True)
    if args.run:
        run(args, output_dir, purification)
    report_path = aggregate(
        output_dir,
        purification,
        append_rows=bool(args.append_rows),
        include_optional_4plus8=bool(args.include_optional_4plus8),
    )
    print(f"Wrote Phase 5 report: {report_path}")


if __name__ == "__main__":
    main()
