from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.evaluation.reproducibility import create_fold_splits, load_folds_json
from src.severstal.rle import rle2mask


@dataclass
class SeverstalSample:
    image_id: str
    image_path: Path
    masks_by_class: dict[int, np.ndarray]
    has_defect: bool
    image: np.ndarray = field(repr=False)


class SeverstalDataset:
    """Severstal steel defect dataset with K-fold cross-validation support."""

    def __init__(
        self,
        data_root: str | Path,
        image_shape: tuple[int, int] = (256, 1600),
        num_classes: int = 4,
        n_folds: int = 5,
        seed: int = 42,
        stratify: bool = True,
        shuffle: bool = True,
        folds_json_path: str | Path | None = None,
    ):
        self.data_root = Path(data_root)
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.n_folds = n_folds
        self.seed = seed
        self.stratify = stratify
        self.shuffle = shuffle

        self.images_dir = self.data_root / "train_images"
        self.annotations_path = self.data_root / "train.csv"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"train_images not found at {self.images_dir}")

        self._image_ids = self._discover_image_ids()
        self._annotations = self._load_annotations()
        self._has_defect = {
            img_id: self._image_has_defect(img_id) for img_id in self._image_ids
        }

        if folds_json_path and Path(folds_json_path).exists():
            self._fold_splits = self._fold_splits_from_json(folds_json_path)
        else:
            labels = [int(self._has_defect[i]) for i in self._image_ids]
            self._fold_splits = create_fold_splits(
                self._image_ids,
                labels,
                n_folds=n_folds,
                seed=seed,
                stratify=stratify,
                shuffle=shuffle,
            )

    def _discover_image_ids(self) -> list[str]:
        ids = sorted(
            p.name for p in self.images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not ids:
            raise FileNotFoundError(f"No images found in {self.images_dir}")
        return ids

    def _load_annotations(self) -> pd.DataFrame:
        if not self.annotations_path.exists():
            return pd.DataFrame(columns=["ImageId", "ClassId", "EncodedPixels"])
        df = pd.read_csv(self.annotations_path)
        df["ImageId"] = df["ImageId"].astype(str)
        df["ClassId"] = df["ClassId"].astype(int)
        return df

    def _image_has_defect(self, image_id: str) -> bool:
        rows = self._annotations[self._annotations["ImageId"] == image_id]
        if rows.empty:
            return False
        for _, row in rows.iterrows():
            rle = row["EncodedPixels"]
            if isinstance(rle, str) and rle.strip():
                return True
            if pd.notna(rle) and str(rle).strip():
                return True
        return False

    def _fold_splits_from_json(
        self, folds_json_path: str | Path
    ) -> list[tuple[list[str], list[str]]]:
        folds_map = load_folds_json(folds_json_path)
        fold_splits: list[tuple[list[str], list[str]]] = []
        for fold_idx in range(self.n_folds):
            val_ids = [
                img_id
                for img_id in self._image_ids
                if folds_map.get(img_id) == fold_idx
            ]
            train_ids = [
                img_id
                for img_id in self._image_ids
                if folds_map.get(img_id) != fold_idx
            ]
            missing = [i for i in self._image_ids if i not in folds_map]
            if missing:
                raise ValueError(f"Images missing from folds.json: {missing[:5]}...")
            fold_splits.append((train_ids, val_ids))
        return fold_splits

    @property
    def image_ids(self) -> list[str]:
        return list(self._image_ids)

    @property
    def fold_splits(self) -> list[tuple[list[str], list[str]]]:
        return self._fold_splits

    def get_fold_split(self, fold_idx: int) -> tuple[list[str], list[str]]:
        return self._fold_splits[fold_idx]

    def get_defect_free_train_ids(self, fold_idx: int) -> list[str]:
        train_ids, _ = self.get_fold_split(fold_idx)
        return [i for i in train_ids if not self._has_defect[i]]

    def _image_has_class(self, image_id: str, class_id: int) -> bool:
        rows = self._annotations[
            (self._annotations["ImageId"] == image_id)
            & (self._annotations["ClassId"] == class_id)
        ]
        for _, row in rows.iterrows():
            rle = row["EncodedPixels"]
            if isinstance(rle, str) and rle.strip():
                return True
            if pd.notna(rle) and str(rle).strip() and str(rle).lower() != "nan":
                return True
        return False

    def get_train_ids_with_class(self, fold_idx: int, class_id: int) -> list[str]:
        """Train-fold image IDs that contain a non-empty mask for the given class."""
        train_ids, _ = self.get_fold_split(fold_idx)
        return sorted(
            i for i in train_ids if self._image_has_class(i, class_id)
        )

    @staticmethod
    def _select_from_class_pool(
        pool: list[str],
        count: int,
        seed: int,
        already_selected: set[str],
    ) -> list[str]:
        """Deterministically pick `count` images from `pool`, skipping duplicates."""
        if count <= 0 or not pool:
            return []

        start = (seed * count) % len(pool)
        selected: list[str] = []
        for offset in range(len(pool)):
            candidate = pool[(start + offset) % len(pool)]
            if candidate in already_selected or candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) >= count:
                break
        return selected

    def get_masks_for_image(self, image_id: str) -> dict[int, np.ndarray]:
        masks = {
            c: np.zeros(self.image_shape, dtype=bool)
            for c in range(1, self.num_classes + 1)
        }
        rows = self._annotations[self._annotations["ImageId"] == image_id]
        for _, row in rows.iterrows():
            class_id = int(row["ClassId"])
            rle = row["EncodedPixels"]
            if pd.isna(rle):
                continue
            rle_str = str(rle).strip()
            if not rle_str or rle_str.lower() == "nan":
                continue
            masks[class_id] |= rle2mask(rle_str, self.image_shape)
        return masks

    def load_image(self, image_id: str) -> np.ndarray:
        path = self.images_dir / image_id
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return image

    def load_sample(self, image_id: str) -> SeverstalSample:
        masks = self.get_masks_for_image(image_id)
        image = self.load_image(image_id)
        return SeverstalSample(
            image_id=image_id,
            image_path=self.images_dir / image_id,
            image=image,
            masks_by_class=masks,
            has_defect=self._has_defect[image_id],
        )

    def select_reference_ids(
        self,
        fold_idx: int,
        shots: int,
        seed: int,
        reference_sampling: str = "class_balanced",
    ) -> list[str]:
        """
        Deterministic k-shot reference image selection from the train fold.

        reference_sampling:
          - ``class_balanced``: ``shots // num_classes`` images per defect class
            (e.g. 8 shots → 2 images with class 1, 2 with class 2, …).
            ``shots`` must be divisible by ``num_classes``.
          - ``defect_free``: legacy mode — defect-free images only.
        """
        if reference_sampling == "defect_free":
            return self._select_defect_free_reference_ids(fold_idx, shots, seed)
        if reference_sampling == "class_balanced":
            return self._select_class_balanced_reference_ids(fold_idx, shots, seed)
        raise ValueError(
            f"Unknown reference_sampling: {reference_sampling!r}. "
            "Choose 'class_balanced' or 'defect_free'."
        )

    def _select_defect_free_reference_ids(
        self,
        fold_idx: int,
        shots: int,
        seed: int,
    ) -> list[str]:
        defect_free = sorted(self.get_defect_free_train_ids(fold_idx))
        if shots == 0:
            return []
        if shots == -1:
            return defect_free
        start = seed * shots
        selected = defect_free[start : start + shots]
        if len(selected) < shots:
            print(
                f"Warning: requested {shots} defect-free reference images but only "
                f"{len(selected)} available in fold {fold_idx}."
            )
        return selected

    def _select_class_balanced_reference_ids(
        self,
        fold_idx: int,
        shots: int,
        seed: int,
    ) -> list[str]:
        if shots == 0:
            return []

        if shots == -1:
            selected_set: set[str] = set()
            for class_id in range(1, self.num_classes + 1):
                selected_set.update(self.get_train_ids_with_class(fold_idx, class_id))
            return sorted(selected_set)

        if shots < 0:
            raise ValueError(f"shots must be non-negative or -1, got {shots}")
        if shots % self.num_classes != 0:
            raise ValueError(
                f"For class_balanced sampling, shots ({shots}) must be divisible "
                f"by num_classes ({self.num_classes}). "
                f"E.g. 4 shots → 1 per class, 8 shots → 2 per class."
            )

        per_class = shots // self.num_classes
        selected: list[str] = []
        already_selected: set[str] = set()

        for class_id in range(1, self.num_classes + 1):
            pool = self.get_train_ids_with_class(fold_idx, class_id)
            class_selected = self._select_from_class_pool(
                pool, per_class, seed + class_id, already_selected
            )
            if len(class_selected) < per_class:
                print(
                    f"Warning: fold {fold_idx}, class {class_id}: requested "
                    f"{per_class} reference images but only {len(class_selected)} "
                    f"available (pool size {len(pool)})."
                )
            for img_id in class_selected:
                already_selected.add(img_id)
            selected.extend(class_selected)

        return selected
