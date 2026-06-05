import numpy as np


def rle2mask(rle: str, shape: tuple[int, int] = (256, 1600)) -> np.ndarray:
    """
    Decode Severstal/Kaggle RLE string to a boolean mask.

    Pixels are numbered top-to-bottom, then left-to-right (Fortran order).
    """
    mask = np.zeros(shape[0] * shape[1], dtype=bool)
    if not isinstance(rle, str) or not rle.strip():
        return mask.reshape(shape, order="F")

    parts = rle.strip().split()
    starts = np.asarray(parts[0::2], dtype=int) - 1
    lengths = np.asarray(parts[1::2], dtype=int)
    for start, length in zip(starts, lengths):
        mask[start : start + length] = True
    return mask.reshape(shape, order="F")


def mask2rle(mask: np.ndarray) -> str:
    """Encode boolean mask to Severstal/Kaggle RLE string."""
    pixels = mask.astype(np.uint8).flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def union_masks(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("At least one mask is required.")
    result = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        result |= m.astype(bool)
    return result
