"""
Pix2Struct extraction module for CAD drawing images.
Uses Hugging Face Pix2Struct (google/pix2struct-base) to extract callouts and embedded dimension labels.
"""
from typing import Dict, Any
from PIL import Image
import torch
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import numpy as np

_pix2struct_cache = {}

def extract_pix2struct(image: np.ndarray, hf_token: str = None) -> Dict[str, Any]:
    """
    Extract callouts and embedded dimension labels from a CAD image using Pix2Struct.
    Args:
        image (np.ndarray): Preprocessed image (grayscale or RGB).
        hf_token (str, optional): Hugging Face token for model loading.
    Returns:
        Dict[str, Any]: Parsed Pix2Struct output (callouts, dimensions, etc.)
    """
    # Convert to PIL Image
    if len(image.shape) == 2:
        pil_img = Image.fromarray(image).convert("RGB")
    else:
        pil_img = Image.fromarray(image)

    # Load processor/model with token (cache for efficiency)
    global _pix2struct_cache
    if hf_token not in _pix2struct_cache:
        processor = Pix2StructProcessor.from_pretrained("google/pix2struct-base", token=hf_token)
        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base", token=hf_token)
        model.eval()
        _pix2struct_cache[hf_token] = (processor, model)
    else:
        processor, model = _pix2struct_cache[hf_token]

    # Prepare input
    prompt = "Extract callouts and dimension labels."
    inputs = processor(pil_img, text=prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    result = processor.decode(outputs[0], skip_special_tokens=True)
    # Try to parse as JSON if possible
    import json
    try:
        parsed = json.loads(result)
    except Exception:
        parsed = {"raw": result}
    return parsed
