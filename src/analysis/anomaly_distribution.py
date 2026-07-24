from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.analysis.adapters import get_dataset_adapter
from src.analysis.aggregation import ScoreAggregator
from src.analysis.config import AnalysisConfig, save_config
from src.analysis.feature_extractors import DinoFeatureExtractor
from src.analysis.label_mapping import map_patch_labels
from src.analysis.plotting import plot_distribution_triptych, save_score_heatmap
from src.analysis.scorers import score_bundle
from src.evaluation.reproducibility import clear_cuda_memory, seed_all


def run_analysis(config: AnalysisConfig) -> Path:
    seed_all(config.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config_used.json")

    extractor = DinoFeatureExtractor(config)
    layer_indices = config.resolve_layer_indices(extractor.num_layers)
    scorer_names = list(config.scorers)

    aggregators: dict[tuple[str, int], ScoreAggregator] = {}
    for scorer_name in scorer_names:
        for layer_idx in layer_indices:
            aggregators[(scorer_name, layer_idx)] = ScoreAggregator(
                scorer_name=scorer_name,
                layer_index=layer_idx,
                save_per_image=config.export.save_per_image,
            )

    samples = list(get_dataset_adapter(config))
    print(f"Running analysis on {len(samples)} images")
    print(f"  Scorers: {scorer_names}")
    print(f"  Layers: {layer_indices}")

    for sample in tqdm(samples, desc="Analysis"):
        native_shape = sample.image.shape[:2]
        bundles = extractor.extract(sample)

        try:
            for layer_idx, bundle in bundles.items():
                labels, coords = map_patch_labels(
                    sample.mask,
                    native_shape,
                    bundle.grid_size,
                    bundle.patch_size,
                    config.model.resolution,
                    rule=config.patch_label.rule,
                    threshold=config.patch_label.threshold,
                )

                for scorer_name in scorer_names:
                    scores = score_bundle(bundle, scorer_name, config)
                    aggregators[(scorer_name, layer_idx)].add(
                        sample.image_id,
                        scores,
                        labels,
                        coords,
                    )

                    if config.export.save_heatmaps:
                        out_dir = (
                            run_dir
                            / scorer_name
                            / f"layer_{layer_idx}"
                            / "heatmaps"
                        )
                        safe_id = sample.image_id.replace("/", "_")
                        save_score_heatmap(
                            sample.image,
                            scores,
                            out_dir / f"{safe_id}.png",
                            title=f"{scorer_name} L{layer_idx}",
                        )
                        del scores
        finally:
            del bundles
            clear_cuda_memory()

    for (scorer_name, layer_idx), aggregator in aggregators.items():
        out_dir = run_dir / scorer_name / f"layer_{layer_idx}"
        aggregator.save(out_dir, save_heatmaps=False)
        plot_distribution_triptych(
            np.asarray(aggregator.healthy_scores, dtype=np.float32),
            np.asarray(aggregator.anomaly_scores, dtype=np.float32),
            scorer_name,
            layer_idx,
            out_dir / "distribution.png",
        )
        summary = aggregator.finalize()
        print(
            f"  {scorer_name}/layer_{layer_idx}: "
            f"healthy={summary['healthy']['count']} "
            f"anomaly={summary['anomaly']['count']} "
            f"AUROC={summary['separability'].get('auroc', 'n/a')}"
        )

    clear_cuda_memory()
    print(f"Results saved to {run_dir}")
    return run_dir
