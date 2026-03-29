"""
LayoutLMv3 extraction module for CAD drawing images.
Uses Hugging Face transformers to extract tokens, bounding boxes, and labels.
"""
from typing import List, Dict
from PIL import Image
import torch
import numpy as np
import pytesseract
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

_layoutlmv3_cache = {}

def normalize_box(box, width, height):
    """Normalize bounding box coordinates to 0–1000 scale."""
    x0, y0, x1, y1 = box
    return [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]

def extract_layoutlmv3(image: np.ndarray, hf_token: str = None) -> List[Dict]:
    """
    Extract tokens, bounding boxes, and labels from a CAD image using LayoutLMv3.
    Args:
        image (np.ndarray): Preprocessed image (grayscale or RGB).
        hf_token (str, optional): Hugging Face token for model loading.
    Returns:
        List[Dict]: List of {text, bbox, label} dicts.
    """
    # Convert to PIL Image
    if len(image.shape) == 2:
        pil_img = Image.fromarray(image).convert("RGB")
    else:
        pil_img = Image.fromarray(image)

    width, height = pil_img.size

    # Run OCR with PyTesseract
    ocr_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    words, boxes = [], []
    for i in range(len(ocr_data["text"])):
        if ocr_data["text"][i].strip() != "":
            words.append(ocr_data["text"][i])
            x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
            boxes.append(normalize_box([x, y, x + w, y + h], width, height))

    # Load processor/model with token (cache for efficiency)
    global _layoutlmv3_cache
    if hf_token not in _layoutlmv3_cache:
        processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False,   # disable built-in OCR since we provide words+boxes
            token=hf_token
        )
        model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", token=hf_token)
        model.eval()
        _layoutlmv3_cache[hf_token] = (processor, model)
    else:
        processor, model = _layoutlmv3_cache[hf_token]

    # Encode image + words + bounding boxes
    encoding = processor(pil_img, words, boxes=boxes, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze())
    boxes = encoding["bbox"].squeeze().tolist()
    labels = [model.config.id2label[pred] for pred in predictions]

    # Build result list
    results = []
    for token, bbox, label in zip(tokens, boxes, labels):
        if token not in processor.tokenizer.all_special_tokens:
            results.append({
                "text": token,
                "bbox": bbox,
                "label": label
            })
    return results
