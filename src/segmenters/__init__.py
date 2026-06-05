from src.segmenters.base import BaseSegmenter, SegmenterOutput, SegmenterPrompts


def build_segmenter(config: dict) -> BaseSegmenter:
    name = config.get("name", "sam2")
    if name == "sam2":
        from src.segmenters.sam2_ultralytics import SAM2Segmenter

        return SAM2Segmenter(
            model_name=config.get("model", "sam2.1_b.pt"),
            device=config.get("device", "cuda:0"),
        )
    raise ValueError(f"Unknown segmenter: {name}")
