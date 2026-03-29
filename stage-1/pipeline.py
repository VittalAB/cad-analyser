"""
Main pipeline for CAD drawing text extraction.
Combines preprocessing, LayoutLMv3, Donut, Pix2Struct, and merge logic.
"""

import cv2
from typing import Dict
import os
from dotenv import load_dotenv
from preprocessing import preprocess_image
from layoutlmv3_extractor import extract_layoutlmv3
from donut_extractor import extract_donut
from pix2struct_extractor import extract_pix2struct
from merge import merge_results
from dotenv import load_dotenv
import os

load_dotenv()  # loads .env automatically

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found. Please set it in your .env file.")

def run_text_extraction_pipeline(image_path: str) -> Dict:
    """
    Run the full text extraction pipeline on a CAD drawing image.
    Args:
        image_path (str): Path to the input image (PNG).
    Returns:
        Dict: Final structured JSON output.
    """
    # Load Hugging Face token from .env
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    # 1. Preprocessing
    processed_img = preprocess_image(image_path)
    image_shape = processed_img.shape

    # 2. LayoutLMv3 extraction
    layoutlmv3_data = extract_layoutlmv3(processed_img, hf_token=hf_token)

    # 3. Donut extraction
    donut_data = extract_donut(processed_img, hf_token=hf_token)

    # 4. Pix2Struct extraction
    pix2struct_data = extract_pix2struct(processed_img, hf_token=hf_token)

    # 5. Merge results
    final_json = merge_results(layoutlmv3_data, donut_data, pix2struct_data, image_shape)
    return final_json


if __name__ == "__main__":
    # Example usage
    image_path = "inputs\\R.jpg"
    result = run_text_extraction_pipeline(image_path)
    save_path = "outputs\\output.json"
    with open(save_path, "w") as f:
        import json
        json.dump(result, f, indent=4)
    print(f"Extraction complete. Output saved to {save_path}")