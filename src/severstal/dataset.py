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
    ) -> list[str]:
        """Deterministic k-shot selection from defect-free train-fold images."""
        defect_free = self.get_defect_free_train_ids(fold_idx)
        defect_free = sorted(defect_free)
        if shots == -1:
            return defect_free
        start = seed * shots
        end = start + shots
        selected = defect_free[start:end]
        if len(selected) < shots and shots > 0:
            print(
                f"Warning: requested {shots} reference images but only "
                f"{len(selected)} available in fold {fold_idx}."
            )
        return selected
