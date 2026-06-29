from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SCORERS = [
    "cls_patch_cosine",
    "patch_l2",
    "sobel_feature",
    "attention_rollout",
]


@dataclass
class ModelConfig:
    name: str = "dinov2_vits14"
    resolution: int = 448


@dataclass
class PatchLabelConfig:
    rule: str = "overlap_ratio_threshold"
    threshold: float = 0.5


@dataclass
class SobelConfig:
    mode: str = "feature"
    norm_reduction: str = "l2"
    image_reduction: str = "mean"


@dataclass
class AttentionRolloutConfig:
    average_heads: bool = True
    include_residual: bool = True
    discard_ratio: float = 0.0
    last_n_layers: int | None = None
    head_reduction: str | None = None


@dataclass
class ExportConfig:
    save_per_image: bool = True
    save_heatmaps: bool = True
    incremental_flush: bool = False


@dataclass
class DatasetConfig:
    name: str = "severstal"
    root: str = "data/severstal"
    image_shape: tuple[int, int] = (256, 1600)
    max_images: int | None = None


@dataclass
class AnalysisConfig:
    seed: int = 42
    device: str = "cuda:0"
    output_dir: str = "results_analysis"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    layers: Any = "last"
    patch_label: PatchLabelConfig = field(default_factory=PatchLabelConfig)
    scorers: list[str] = field(default_factory=lambda: list(DEFAULT_SCORERS))
    sobel: SobelConfig = field(default_factory=SobelConfig)
    attention_rollout: AttentionRolloutConfig = field(
        default_factory=AttentionRolloutConfig
    )
    export: ExportConfig = field(default_factory=ExportConfig)

    def resolve_layer_indices(self, num_layers: int) -> list[int]:
        if self.layers == "last":
            return [num_layers - 1]
        if self.layers == "all":
            return list(range(num_layers))
        if isinstance(self.layers, int):
            return [self.layers]
        if isinstance(self.layers, list):
            return [int(i) for i in self.layers]
        raise ValueError(f"Invalid layers config: {self.layers!r}")


def _dict_to_config(data: dict) -> AnalysisConfig:
    model = ModelConfig(**data.get("model", {}))
    dataset_raw = data.get("dataset", {})
    image_shape = dataset_raw.get("image_shape", [256, 1600])
    dataset = DatasetConfig(
        name=dataset_raw.get("name", "severstal"),
        root=dataset_raw.get("root", "data/severstal"),
        image_shape=tuple(image_shape),
        max_images=dataset_raw.get("max_images"),
    )
    patch_label = PatchLabelConfig(**data.get("patch_label", {}))
    sobel = SobelConfig(**data.get("sobel", {}))
    attention_rollout = AttentionRolloutConfig(**data.get("attention_rollout", {}))
    export = ExportConfig(**data.get("export", {}))
    scorers = data.get("scorers", list(DEFAULT_SCORERS))
    return AnalysisConfig(
        seed=data.get("seed", 42),
        device=data.get("device", "cuda:0"),
        output_dir=data.get("output_dir", "results_analysis"),
        model=model,
        dataset=dataset,
        layers=data.get("layers", "last"),
        patch_label=patch_label,
        scorers=scorers,
        sobel=sobel,
        attention_rollout=attention_rollout,
        export=export,
    )


def load_config(path: str | Path) -> AnalysisConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _dict_to_config(data)


def save_config(config: AnalysisConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _serialize(v) for k, v in asdict(obj).items()}
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_serialize(config), f, indent=2)
