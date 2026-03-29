"""
Donut extraction module for CAD drawing images.
Uses Hugging Face Donut (naver-clova-ix/donut-base) for OCR-free extraction of title block fields, BOM tables, and revision tables.
"""
from typing import Dict
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
import numpy as np

_donut_cache = {}

def extract_donut(image: np.ndarray, hf_token: str = None) -> Dict:
    """
    Extract title block fields, BOM tables, and revision tables from a CAD image using Donut.
    Args:
        image (np.ndarray): Preprocessed image (grayscale or RGB).
        hf_token (str, optional): Hugging Face token for model loading.
    Returns:
        Dict: Parsed Donut output (fields, tables, etc.)
    """
    # Convert to PIL Image
    if len(image.shape) == 2:
        pil_img = Image.fromarray(image).convert("RGB")
    else:
        pil_img = Image.fromarray(image)

    # Load processor/model with token (cache for efficiency)
    global _donut_cache
    if hf_token not in _donut_cache:
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base", token=hf_token)
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", token=hf_token)
        model.eval()
        _donut_cache[hf_token] = (processor, model)
    else:
        processor, model = _donut_cache[hf_token]

    # Prepare input
    task_prompt = "<s_cad-drawing>"
    pixel_values = processor(pil_img, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = model.generate(pixel_values, max_length=1024)
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    # Try to parse as JSON if possible
    import json
    try:
        parsed = json.loads(result)
    except Exception:
        parsed = {"raw": result}
    return parsed
