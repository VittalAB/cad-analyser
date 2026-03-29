"""
Merge logic for CAD drawing text extraction pipeline.
Combines results from LayoutLMv3, Donut, and Pix2Struct modules into a structured JSON output.
"""
import re
from typing import List, Dict, Any

def is_dimension(text: str) -> bool:
    """Identify dimension-like tokens using regex."""
    pattern = r"(\b\d+(\.\d+)?\b|Ø\s*\d+|R\s*\d+|±\s*\d+|\d+°)"
    return bool(re.search(pattern, text))

def normalize_bbox(bbox, width, height):
    """Normalize bounding box to 0–1000 coordinate space."""
    x0, y0, x1, y1 = bbox
    return [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]

def merge_results(layoutlmv3_data: List[Dict], donut_data: Dict, pix2struct_data: Dict, image_shape) -> Dict[str, Any]:
    """
    Merge results from all extractors into a structured JSON.
    Args:
        layoutlmv3_data (List[Dict]): Output from LayoutLMv3 extractor.
        donut_data (Dict): Output from Donut extractor.
        pix2struct_data (Dict): Output from Pix2Struct extractor.
        image_shape (tuple): (height, width) of the image.
    Returns:
        Dict[str, Any]: Final structured JSON.
    """
    height, width = image_shape
    dimensions = []
    annotations = []
    callouts = []
    all_texts = set()

    # LayoutLMv3: dimensions, annotations
    for item in layoutlmv3_data:
        text = item["text"]
        bbox = normalize_bbox(item["bbox"], width, height)
        label = item["label"]
        if is_dimension(text):
            if text not in all_texts:
                dimensions.append({"text": text, "bbox": bbox, "label": label})
                all_texts.add(text)
        else:
            if text not in all_texts:
                annotations.append({"text": text, "bbox": bbox, "label": label})
                all_texts.add(text)

    # Pix2Struct: callouts, embedded dimensions
    if "callouts" in pix2struct_data:
        for callout in pix2struct_data["callouts"]:
            if callout not in all_texts:
                callouts.append(callout)
                all_texts.add(callout)
    if "dimensions" in pix2struct_data:
        for dim in pix2struct_data["dimensions"]:
            if dim not in all_texts:
                dimensions.append({"text": dim, "bbox": None, "label": "pix2struct"})
                all_texts.add(dim)

    # Donut: title block, tables
    title_block = donut_data.get("title_block", donut_data.get("fields", {}))
    tables = {
        "bom": donut_data.get("bom_table", donut_data.get("bom", {})),
        "revision": donut_data.get("revision_table", donut_data.get("revision", {})),
    }

    # Fallback for raw Donut output
    if not title_block and "raw" in donut_data:
        title_block = {"raw": donut_data["raw"]}

    # Fallback for raw Pix2Struct output
    if not callouts and "raw" in pix2struct_data:
        callouts = [pix2struct_data["raw"]]

    return {
        "dimensions": dimensions,
        "annotations": annotations,
        "callouts": callouts,
        "title_block": title_block,
        "tables": tables,
    }
