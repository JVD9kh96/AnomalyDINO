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
        self._classes_by_image = self._index_classes_by_image()
        self._has_defect = {
            image_id: bool(self._classes_by_image[image_id])
            for image_id in self._image_ids
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

    @staticmethod
    def _has_nonempty_rle(rle: object) -> bool:
        if pd.isna(rle):
            return False
        value = str(rle).strip()
        return bool(value) and value.lower() != "nan"

    def _index_classes_by_image(self) -> dict[str, set[int]]:
        """Index annotations once instead of rescanning the table per image."""
        classes_by_image = {image_id: set() for image_id in self._image_ids}
        for row in self._annotations.itertuples(index=False):
            image_id = str(row.ImageId)
            if image_id not in classes_by_image:
                continue
            if self._has_nonempty_rle(row.EncodedPixels):
                classes_by_image[image_id].add(int(row.ClassId))
        return classes_by_image

    def _image_has_defect(self, image_id: str) -> bool:
        return bool(self._classes_by_image.get(image_id, set()))

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

    def image_has_defect(self, image_id: str) -> bool:
        """Return the persisted image-level defect status for ``image_id``."""
        if image_id not in self._has_defect:
            raise KeyError(f"Unknown image ID: {image_id}")
        return bool(self._has_defect[image_id])

    def get_image_classes(self, image_id: str) -> list[int]:
        """Return sorted non-empty defect classes present in ``image_id``."""
        if image_id not in self._has_defect:
            raise KeyError(f"Unknown image ID: {image_id}")
        return sorted(self._classes_by_image[image_id])

    def _image_has_class(self, image_id: str, class_id: int) -> bool:
        return class_id in self._classes_by_image.get(image_id, set())

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
        selected = self._select_from_sorted_pool(defect_free, shots, seed)
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

    def get_image_classes(self, image_id: str) -> list[int]:
        """Return sorted defect class IDs present on an image (empty if clean)."""
        classes: list[int] = []
        for class_id in range(1, self.num_classes + 1):
            if self._image_has_class(image_id, class_id):
                classes.append(class_id)
        return classes

    def reference_image_metadata(
        self, image_ids: list[str]
    ) -> tuple[dict[str, bool], dict[str, list[int]]]:
        has_defect = {i: bool(self._has_defect[i]) for i in image_ids}
        classes = {i: self.get_image_classes(i) for i in image_ids}
        return has_defect, classes

    def _select_from_sorted_pool(
        self,
        pool: list[str],
        count: int,
        seed: int,
        exclude_ids: set[str] | None = None,
    ) -> list[str]:
        """Deterministic contiguous / wrap selection from a sorted pool."""
        if count <= 0:
            return []
        exclude = exclude_ids or set()
        filtered = [i for i in sorted(pool) if i not in exclude]
        if not filtered:
            return []
        if count >= len(filtered):
            return list(filtered)
        start = (seed * count) % len(filtered)
        selected: list[str] = []
        for offset in range(len(filtered)):
            candidate = filtered[(start + offset) % len(filtered)]
            if candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) >= count:
                break
        return selected

    def select_additional_reference_ids(
        self,
        fold_idx: int,
        n: int,
        seed: int,
        sampling: str = "class_balanced",
        exclude_ids: set[str] | list[str] | None = None,
    ) -> list[str]:
        """
        Select additional reference images from the train fold.

        sampling:
          - ``class_balanced``: even counts across defect classes
          - ``random_train``: deterministic sample from all train IDs
          - ``mixed``: same as random_train (unverified mixed train images)
        """
        exclude = set(exclude_ids or [])
        train_ids, _ = self.get_fold_split(fold_idx)

        if n <= 0:
            return []

        if sampling == "class_balanced":
            # Reuse class-balanced logic then drop excludes / top-up if needed.
            raw = self._select_class_balanced_reference_ids(fold_idx, n, seed)
            selected = [i for i in raw if i not in exclude]
            if len(selected) < n:
                # Fill remaining from defect train pool excluding already chosen.
                defect_pool = sorted(
                    i for i in train_ids if self._has_defect[i] and i not in exclude
                )
                already = set(selected)
                for img_id in self._select_from_sorted_pool(
                    defect_pool, n - len(selected), seed + 17, already
                ):
                    selected.append(img_id)
            return selected[:n]

        if sampling in ("random_train", "mixed"):
            return self._select_from_sorted_pool(train_ids, n, seed, exclude)

        raise ValueError(
            f"Unknown additional_sampling: {sampling!r}. "
            "Choose 'class_balanced', 'random_train', or 'mixed'."
        )

    def select_reference_composition(
        self,
        fold_idx: int,
        seed: int,
        *,
        reference_mode: str = "clean",
        clean_shots: int = 2,
        additional_shots: int = 0,
        additional_sampling: str = "class_balanced",
    ) -> dict:
        """
        Select clean + additional reference IDs for a reference_mode study.

        Returns a dict with IDs and metadata suitable for fold metrics JSON.
        """
        train_ids, val_ids = self.get_fold_split(fold_idx)
        val_set = set(val_ids)

        if reference_mode == "class_balanced_all":
            clean_ids: list[str] = []
            if clean_shots > 0:
                clean_ids = self._select_defect_free_reference_ids(
                    fold_idx, clean_shots, seed
                )
            n_additional = additional_shots if additional_shots > 0 else 8
            additional_ids = self.select_additional_reference_ids(
                fold_idx,
                n_additional,
                seed + 1,
                sampling="class_balanced",
                exclude_ids=set(clean_ids),
            )
        elif reference_mode in (
            "clean",
            "contaminated_all",
            "oracle_purified",
            "auto_purified",
            "random_filtered",
            "fixed_ratio_trim",
        ):
            clean_ids = self._select_defect_free_reference_ids(
                fold_idx, clean_shots, seed
            )
            if reference_mode == "clean":
                additional_ids = []
            else:
                additional_ids = self.select_additional_reference_ids(
                    fold_idx,
                    additional_shots,
                    seed + 1,
                    sampling=additional_sampling,
                    exclude_ids=set(clean_ids),
                )
        else:
            raise ValueError(
                f"Unknown reference_mode: {reference_mode!r}. "
                "Choose clean, contaminated_all, class_balanced_all, "
                "oracle_purified, auto_purified, random_filtered, or fixed_ratio_trim."
            )

        all_ref_ids = list(dict.fromkeys([*clean_ids, *additional_ids]))
        overlap = [i for i in all_ref_ids if i in val_set]
        if overlap:
            raise RuntimeError(
                f"Reference IDs overlap validation fold: {overlap[:5]}"
            )
        missing = [i for i in all_ref_ids if i not in set(train_ids)]
        if missing:
            raise RuntimeError(
                f"Reference IDs not in train fold: {missing[:5]}"
            )

        has_defect, classes = self.reference_image_metadata(all_ref_ids)
        return {
            "reference_mode": reference_mode,
            "clean_reference_ids": clean_ids,
            "additional_reference_ids": additional_ids,
            "reference_image_has_defect": has_defect,
            "reference_classes": classes,
            "n_memory_patches_before_filtering": 0,
            "n_memory_patches_after_filtering": 0,
        }
