from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from src.analysis.types import AnalysisSample
from src.severstal.dataset import SeverstalDataset
from src.severstal.rle import union_masks


class DatasetAdapter(Protocol):
    def __iter__(self) -> Iterator[AnalysisSample]: ...


def iter_severstal_samples(
    data_root: str,
    image_shape: tuple[int, int] = (256, 1600),
    image_ids: list[str] | None = None,
    max_images: int | None = None,
) -> Iterator[AnalysisSample]:
    dataset = SeverstalDataset(data_root=data_root, image_shape=image_shape)
    ids = image_ids if image_ids is not None else dataset.image_ids
    if max_images is not None:
        ids = ids[:max_images]

    for image_id in ids:
        sample = dataset.load_sample(image_id)
        mask = union_masks(list(sample.masks_by_class.values()))
        yield AnalysisSample(
            image_id=sample.image_id,
            image=sample.image,
            mask=mask,
            meta={"has_defect": sample.has_defect},
        )


def get_dataset_adapter(config) -> Iterator[AnalysisSample]:
    name = config.dataset.name
    if name == "severstal":
        return iter_severstal_samples(
            data_root=config.dataset.root,
            image_shape=config.dataset.image_shape,
            max_images=config.dataset.max_images,
        )
    raise ValueError(f"Unknown dataset adapter: {name!r}")
