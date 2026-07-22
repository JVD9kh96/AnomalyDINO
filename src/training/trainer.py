from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones import get_model
from src.evaluation.reproducibility import save_json
from src.severstal.dataset import SeverstalSample
from src.training.data import (
    PatchImageDataset,
    collate_patch_images,
    flatten_batch_targets,
)
from src.training.heads import PatchLinearClassifier
from src.training.logging_utils import save_history
from src.training.losses import binary_loss, multiclass_ce_loss
from src.training.plots import save_training_curves_pdf


@dataclass
class TrainConfig:
    classification_mode: str = "binary"
    binary_loss: str = "bce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    lambda_mc: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    epochs: int = 50
    num_workers: int = 0
    num_classes: int = 4
    resolution: int = 448
    gt_overlap_threshold: float = 0.5
    threshold_num_steps: int = 101
    save_plots: bool = True
    log_format: str = "both"
    seed: int = 42
    device: str = "cuda:0"
    model_name: str = "dinov2_vits14"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dicts(
        cls,
        detector_cfg: dict[str, Any],
        train_cfg: dict[str, Any] | None = None,
        patch_eval: dict[str, Any] | None = None,
        data_cfg: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> TrainConfig:
        train_cfg = train_cfg or {}
        patch_eval = patch_eval or {}
        data_cfg = data_cfg or {}
        return cls(
            classification_mode=detector_cfg.get(
                "classification_mode", "binary"
            ),
            binary_loss=detector_cfg.get("binary_loss", "bce"),
            focal_gamma=float(detector_cfg.get("focal_gamma", 2.0)),
            focal_alpha=float(detector_cfg.get("focal_alpha", 0.25)),
            lambda_mc=float(detector_cfg.get("lambda_mc", 1.0)),
            lr=float(detector_cfg.get("lr", 1e-3)),
            weight_decay=float(detector_cfg.get("weight_decay", 1e-4)),
            batch_size=int(detector_cfg.get("batch_size", 4)),
            epochs=int(detector_cfg.get("epochs", 50)),
            num_workers=int(detector_cfg.get("num_workers", 0)),
            num_classes=int(data_cfg.get("num_classes", 4)),
            resolution=int(detector_cfg.get("resolution", 448)),
            gt_overlap_threshold=float(
                patch_eval.get("gt_overlap_threshold", 0.5)
            ),
            threshold_num_steps=int(
                detector_cfg.get("threshold_num_steps", 101)
            ),
            save_plots=bool(train_cfg.get("save_plots", True)),
            log_format=str(train_cfg.get("log_format", "both")),
            seed=seed,
            device=str(detector_cfg.get("device", "cuda:0")),
            model_name=str(detector_cfg.get("model_name", "dinov2_vits14")),
        )


def _metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if (
        np.isfinite(precision)
        and np.isfinite(recall)
        and (precision + recall) > 0
    ):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _binary_metrics_at_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    pred = scores >= threshold
    gt = targets.astype(bool)
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = int(np.sum(~pred & ~gt))
    return _metrics_from_counts(tp, fp, fn, tn)


def find_optimal_f1_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    num_steps: int = 101,
) -> tuple[float, dict[str, float]]:
    """Sweep thresholds in [0, 1] and return the one with best F1."""
    if scores.size == 0:
        return 0.5, _metrics_from_counts(0, 0, 0, 0)

    thresholds = np.linspace(0.0, 1.0, num_steps)
    best_t = 0.5
    best_metrics = _binary_metrics_at_threshold(scores, targets, best_t)
    best_f1 = best_metrics["f1"]
    if not np.isfinite(best_f1):
        best_f1 = -1.0

    for t in thresholds:
        metrics = _binary_metrics_at_threshold(scores, targets, float(t))
        f1 = metrics["f1"]
        if np.isfinite(f1) and f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
            best_metrics = metrics
    return best_t, best_metrics


class PatchClassifierTrainer:
    """Train linear probe(s) on frozen DINOv2 patch tokens."""

    def __init__(self, config: TrainConfig):
        if config.classification_mode not in ("binary", "binary_multiclass"):
            raise ValueError(
                f"classification_mode must be 'binary' or 'binary_multiclass', "
                f"got {config.classification_mode!r}."
            )
        self.config = config
        self.device = torch.device(
            "cuda" if "cuda" in config.device and torch.cuda.is_available() else "cpu"
        )
        self._backbone = None
        self._patch_size = 14
        self.classifier: PatchLinearClassifier | None = None
        self.optimal_threshold: float = 0.5
        self.history: list[dict[str, Any]] = []

    def _ensure_backbone(self) -> None:
        if self._backbone is not None:
            return
        if "cuda" in self.config.device:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.config.device[-1]))
        self._backbone = get_model(
            self.config.model_name,
            str(self.device),
            self.config.resolution,
        )
        self._backbone.model.eval()
        for p in self._backbone.model.parameters():
            p.requires_grad_(False)
        self._patch_size = getattr(self._backbone.model, "patch_size", 14)

    def _infer_feature_dim(self, sample_image: np.ndarray) -> int:
        self._ensure_backbone()
        assert self._backbone is not None
        tensor, _ = self._backbone.prepare_image(sample_image)
        with torch.inference_mode():
            feats = self._backbone.extract_features(tensor)
        return int(feats.shape[-1])

    def _extract_patch_tokens_batch(
        self, images: list[np.ndarray]
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Extract frozen patch tokens for a list of images; returns (N_total, D)."""
        self._ensure_backbone()
        assert self._backbone is not None

        tokens_list: list[torch.Tensor] = []
        grid_sizes: list[tuple[int, int]] = []
        # no_grad (not inference_mode) so tensors can feed the trainable head
        with torch.no_grad():
            for image in images:
                tensor, grid_size = self._backbone.prepare_image(image)
                batch = tensor.unsqueeze(0).to(self.device)
                out = self._backbone.model.get_intermediate_layers(
                    batch, n=1, return_class_token=False, norm=True
                )[0]
                # out: (1, N_patches, D) — clone so Linear can attach grads to weights
                tokens_list.append(out.squeeze(0).clone())
                grid_sizes.append(grid_size)
        return torch.cat(tokens_list, dim=0), grid_sizes

    def _build_loader(
        self, samples: Sequence[SeverstalSample], shuffle: bool
    ) -> DataLoader:
        dataset = PatchImageDataset(
            samples,
            resolution=self.config.resolution,
            patch_size=self._patch_size,
            gt_overlap_threshold=self.config.gt_overlap_threshold,
            num_classes=self.config.num_classes,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_patch_images,
        )

    def _compute_batch_loss(
        self,
        binary_logits: torch.Tensor,
        multiclass_logits: torch.Tensor | None,
        binary_targets: torch.Tensor,
        multiclass_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config
        b_loss = binary_loss(
            binary_logits,
            binary_targets,
            loss_type=cfg.binary_loss,
            focal_gamma=cfg.focal_gamma,
            focal_alpha=cfg.focal_alpha,
        )
        stats: dict[str, float] = {"binary_loss": float(b_loss.detach().cpu())}
        total = b_loss

        if (
            cfg.classification_mode == "binary_multiclass"
            and multiclass_logits is not None
        ):
            # GT-anomalous with a valid class label (overlap may split across classes)
            anomalous = (binary_targets > 0.5) & (multiclass_targets >= 0)
            if anomalous.any():
                mc_loss = multiclass_ce_loss(
                    multiclass_logits[anomalous],
                    multiclass_targets[anomalous],
                )
            else:
                mc_loss = binary_logits.new_zeros(())
            total = total + cfg.lambda_mc * mc_loss
            stats["multiclass_loss"] = float(mc_loss.detach().cpu())
        else:
            stats["multiclass_loss"] = float("nan")

        stats["loss"] = float(total.detach().cpu())
        return total, stats

    @torch.no_grad()
    def _collect_predictions(
        self, samples: Sequence[SeverstalSample]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return flat binary scores/targets and multiclass preds/targets (anom only)."""
        assert self.classifier is not None
        self.classifier.eval()
        loader = self._build_loader(samples, shuffle=False)

        all_scores: list[np.ndarray] = []
        all_binary: list[np.ndarray] = []
        all_mc_pred: list[np.ndarray] = []
        all_mc_tgt: list[np.ndarray] = []

        for batch in loader:
            tokens, _ = self._extract_patch_tokens_batch(batch["images"])
            binary_tgt, mc_tgt = flatten_batch_targets(
                batch["binary"], batch["multiclass"]
            )
            binary_tgt = binary_tgt.to(self.device)
            mc_tgt = mc_tgt.to(self.device)
            binary_logits, mc_logits = self.classifier(tokens)
            scores = torch.sigmoid(binary_logits).cpu().numpy()
            all_scores.append(scores)
            all_binary.append(binary_tgt.cpu().numpy())

            if mc_logits is not None:
                anomalous = (binary_tgt > 0.5) & (mc_tgt >= 0)
                if anomalous.any():
                    pred = mc_logits[anomalous].argmax(dim=-1).cpu().numpy()
                    tgt = mc_tgt[anomalous].cpu().numpy()
                    all_mc_pred.append(pred)
                    all_mc_tgt.append(tgt)

        scores_np = (
            np.concatenate(all_scores) if all_scores else np.zeros(0, dtype=np.float32)
        )
        binary_np = (
            np.concatenate(all_binary) if all_binary else np.zeros(0, dtype=np.float32)
        )
        mc_pred_np = (
            np.concatenate(all_mc_pred) if all_mc_pred else np.zeros(0, dtype=np.int64)
        )
        mc_tgt_np = (
            np.concatenate(all_mc_tgt) if all_mc_tgt else np.zeros(0, dtype=np.int64)
        )
        return scores_np, binary_np, mc_pred_np, mc_tgt_np

    def _eval_split(
        self,
        samples: Sequence[SeverstalSample],
        threshold: float,
        prefix: str,
    ) -> dict[str, float]:
        if not samples:
            return {
                f"{prefix}_loss": float("nan"),
                f"{prefix}_binary_loss": float("nan"),
                f"{prefix}_multiclass_loss": float("nan"),
                f"{prefix}_binary_precision": float("nan"),
                f"{prefix}_binary_recall": float("nan"),
                f"{prefix}_binary_f1": float("nan"),
                f"{prefix}_multiclass_acc": float("nan"),
            }

        assert self.classifier is not None
        self.classifier.eval()
        loader = self._build_loader(samples, shuffle=False)

        total_loss = 0.0
        total_b = 0.0
        total_mc = 0.0
        n_batches = 0
        all_scores: list[np.ndarray] = []
        all_binary: list[np.ndarray] = []
        mc_correct = 0
        mc_total = 0

        with torch.no_grad():
            for batch in loader:
                tokens, _ = self._extract_patch_tokens_batch(batch["images"])
                binary_tgt, mc_tgt = flatten_batch_targets(
                    batch["binary"], batch["multiclass"]
                )
                binary_tgt = binary_tgt.to(self.device)
                mc_tgt = mc_tgt.to(self.device)
                binary_logits, mc_logits = self.classifier(tokens)
                _, stats = self._compute_batch_loss(
                    binary_logits, mc_logits, binary_tgt, mc_tgt
                )
                total_loss += stats["loss"]
                total_b += stats["binary_loss"]
                if np.isfinite(stats["multiclass_loss"]):
                    total_mc += stats["multiclass_loss"]
                n_batches += 1

                scores = torch.sigmoid(binary_logits).cpu().numpy()
                all_scores.append(scores)
                all_binary.append(binary_tgt.cpu().numpy())

                if mc_logits is not None:
                    anomalous = (binary_tgt > 0.5) & (mc_tgt >= 0)
                    if anomalous.any():
                        pred = mc_logits[anomalous].argmax(dim=-1)
                        tgt = mc_tgt[anomalous]
                        mc_correct += int((pred == tgt).sum().cpu())
                        mc_total += int(anomalous.sum().cpu())

        scores_np = np.concatenate(all_scores) if all_scores else np.zeros(0)
        binary_np = np.concatenate(all_binary) if all_binary else np.zeros(0)
        bin_m = _binary_metrics_at_threshold(scores_np, binary_np, threshold)
        mc_acc = (mc_correct / mc_total) if mc_total > 0 else float("nan")

        return {
            f"{prefix}_loss": total_loss / max(n_batches, 1),
            f"{prefix}_binary_loss": total_b / max(n_batches, 1),
            f"{prefix}_multiclass_loss": (
                total_mc / max(n_batches, 1)
                if self.config.classification_mode == "binary_multiclass"
                else float("nan")
            ),
            f"{prefix}_binary_precision": bin_m["precision"],
            f"{prefix}_binary_recall": bin_m["recall"],
            f"{prefix}_binary_f1": bin_m["f1"],
            f"{prefix}_multiclass_acc": mc_acc,
        }

    def fit(
        self,
        train_samples: Sequence[SeverstalSample],
        val_samples: Sequence[SeverstalSample] | None = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        if not train_samples:
            raise ValueError("train_samples must be non-empty for supervised training.")

        val_samples = val_samples or []
        output_dir = Path(output_dir) if output_dir is not None else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_backbone()
        feature_dim = self._infer_feature_dim(train_samples[0].image)
        self.classifier = PatchLinearClassifier(
            feature_dim=feature_dim,
            num_classes=self.config.num_classes,
            classification_mode=self.config.classification_mode,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        train_loader = self._build_loader(train_samples, shuffle=True)
        self.history = []
        # Use 0.5 during epoch metrics; refine after training on val
        running_threshold = 0.5

        for epoch in range(1, self.config.epochs + 1):
            self.classifier.train()
            epoch_loss = 0.0
            epoch_b = 0.0
            epoch_mc = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")
            for batch in pbar:
                tokens, _ = self._extract_patch_tokens_batch(batch["images"])
                # Tokens are inference_mode; re-enable grads path for head only
                tokens = tokens.detach()
                binary_tgt, mc_tgt = flatten_batch_targets(
                    batch["binary"], batch["multiclass"]
                )
                binary_tgt = binary_tgt.to(self.device)
                mc_tgt = mc_tgt.to(self.device)

                binary_logits, mc_logits = self.classifier(tokens)
                loss, stats = self._compute_batch_loss(
                    binary_logits, mc_logits, binary_tgt, mc_tgt
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += stats["loss"]
                epoch_b += stats["binary_loss"]
                if np.isfinite(stats["multiclass_loss"]):
                    epoch_mc += stats["multiclass_loss"]
                n_batches += 1
                pbar.set_postfix(loss=f"{stats['loss']:.4f}")

            row: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(n_batches, 1),
                "train_binary_loss": epoch_b / max(n_batches, 1),
                "train_multiclass_loss": (
                    epoch_mc / max(n_batches, 1)
                    if self.config.classification_mode == "binary_multiclass"
                    else float("nan")
                ),
            }
            # Train metrics at current threshold
            train_metrics = self._eval_split(
                train_samples, running_threshold, prefix="train"
            )
            # Avoid double-counting loss keys from _eval_split train_loss
            for k, v in train_metrics.items():
                if k.endswith("_loss"):
                    continue
                row[k] = v

            if val_samples:
                val_metrics = self._eval_split(
                    val_samples, running_threshold, prefix="val"
                )
                row.update(val_metrics)

            self.history.append(row)
            if output_dir is not None:
                save_history(self.history, output_dir, self.config.log_format)

        # Optimal F1 threshold on validation (fallback to train if no val)
        thresh_samples = val_samples if val_samples else train_samples
        scores, targets, _, _ = self._collect_predictions(thresh_samples)
        self.optimal_threshold, thresh_metrics = find_optimal_f1_threshold(
            scores, targets, num_steps=self.config.threshold_num_steps
        )

        result = {
            "optimal_threshold": self.optimal_threshold,
            "threshold_metrics": thresh_metrics,
            "feature_dim": feature_dim,
            "epochs": self.config.epochs,
            "classification_mode": self.config.classification_mode,
            "n_train": len(train_samples),
            "n_val": len(val_samples),
        }

        if output_dir is not None:
            save_history(self.history, output_dir, self.config.log_format)
            if self.config.save_plots:
                save_training_curves_pdf(
                    self.history, output_dir / "training_curves.pdf"
                )
            save_json(
                {
                    "optimal_threshold": self.optimal_threshold,
                    "threshold_metrics": thresh_metrics,
                },
                output_dir / "threshold.json",
            )
            self.save_checkpoint(output_dir / "checkpoint.pt", meta=result)

        return result

    def save_checkpoint(
        self,
        path: str | Path,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if self.classifier is None:
            raise RuntimeError("No classifier to save; call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "classifier_state_dict": self.classifier.state_dict(),
            "feature_dim": self.classifier.feature_dim,
            "num_classes": self.classifier.num_classes,
            "classification_mode": self.classifier.classification_mode,
            "optimal_threshold": self.optimal_threshold,
            "config": asdict(self.config),
            "meta": meta or {},
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        self.classifier = PatchLinearClassifier(
            feature_dim=int(payload["feature_dim"]),
            num_classes=int(payload["num_classes"]),
            classification_mode=str(payload["classification_mode"]),
        ).to(self.device)
        self.classifier.load_state_dict(payload["classifier_state_dict"])
        self.classifier.eval()
        self.optimal_threshold = float(payload.get("optimal_threshold", 0.5))
        self._ensure_backbone()

    @torch.no_grad()
    def predict_image(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int]]:
        """
        Returns:
            patch_scores: (H, W) binary probabilities
            patch_class_scores: (H, W, C) gated class scores or None
            grid_size: (H, W)
        """
        if self.classifier is None:
            raise RuntimeError("Classifier not initialized; call fit() or load_checkpoint().")
        self.classifier.eval()
        tokens, grid_sizes = self._extract_patch_tokens_batch([image])
        grid_size = grid_sizes[0]
        binary_logits, mc_logits = self.classifier(tokens)
        scores = torch.sigmoid(binary_logits).cpu().numpy().reshape(grid_size)

        class_scores = None
        if mc_logits is not None:
            probs = torch.softmax(mc_logits, dim=-1).cpu().numpy().reshape(
                *grid_size, -1
            )
            pred_class = probs.argmax(axis=-1)
            anomalous = scores >= self.optimal_threshold
            class_scores = np.zeros_like(probs, dtype=np.float32)
            # Gate: only predicted class gets the binary score when anomalous
            for c in range(probs.shape[-1]):
                mask = anomalous & (pred_class == c)
                class_scores[..., c][mask] = scores[mask]

        return scores.astype(np.float32), class_scores, grid_size
