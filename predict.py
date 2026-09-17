#!/usr/bin/env python
"""
DocVisionAI: Single-Sample Inference CLI.
Processes a document image or PDF end-to-end and outputs structured JSON.
"""
import os
import sys
import json
import argparse
import logging

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.preprocessing.image_processor import ImagePreprocessor
from app.ocr.ocr_engine import OCREngine
from app.models.vlm_model import DocVisionVLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DocVisionAI.PredictCLI")

def main():
    parser = argparse.ArgumentParser(description="DocVisionAI Document Extraction CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or PDF file")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON file. Prints to stdout if omitted.")
    parser.add_argument("--doc_type", type=str, default="auto", choices=["auto", "invoice", "aadhaar", "pan"], help="Document schema type")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # 1. Initialize Engines
    preprocessor = ImagePreprocessor()
    ocr = OCREngine()
    vlm = DocVisionVLM()

    temp_image_path = args.input
    is_pdf = args.input.lower().endswith(".pdf")
    
    # 2. PDF Conversion if needed
    if is_pdf:
        logger.info("Input file is a PDF. Converting first page to image...")
        temp_img_dir = os.path.dirname(os.path.abspath(args.input))
        converted_img_path = os.path.join(temp_img_dir, f"temp_pdf_page_{int(time.time())}.jpg")
        try:
            temp_image_path = ocr.convert_pdf_to_image(args.input, converted_img_path)
        except Exception as e:
            logger.error(f"Failed to convert PDF: {e}")
            sys.exit(1)

    # 3. Preprocess Image
    preprocessed_path = temp_image_path + ".preprocessed.jpg"
    try:
        preprocessor.preprocess_image(temp_image_path, preprocessed_path)
    except Exception as e:
        logger.warning(f"Preprocessing encountered an issue: {e}. Proceeding with raw image.")
        preprocessed_path = temp_image_path

    # 4. Run OCR
    logger.info("Executing OCR text extraction...")
    ocr_results, detected_doc_type = ocr.extract_text(preprocessed_path, args.doc_type)
    
    # 5. Run VLM Inference
    logger.info(f"Executing VLM Extraction for schema type '{detected_doc_type}'...")
    result = vlm.process_document(preprocessed_path, ocr_results, detected_doc_type)

    # Cleanup temp preprocessed image if we created it
    if os.path.exists(preprocessed_path) and preprocessed_path != temp_image_path:
        try:
            os.remove(preprocessed_path)
        except OSError:
            pass
            
    # Cleanup temp PDF converted image if we created it
    if is_pdf and os.path.exists(temp_image_path):
        try:
            os.remove(temp_image_path)
        except OSError:
            pass

    # 6. Format and Output
    output_payload = {
        "success": True,
        "input_file": os.path.basename(args.input),
        "document_type": result.get("document_type"),
        "fields": result.get("fields"),
        "confidence": result.get("confidence"),
        "metrics": {
            "ocr_boxes": len(ocr_results),
            "vlm_latency_sec": result.get("inference_latency_sec")
        }
    }

    formatted_json = json.dumps(output_payload, indent=2)
    
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(formatted_json)
        logger.info(f"Structured extraction saved to: {args.output}")
    else:
        print(formatted_json)

if __name__ == "__main__":
    import time
    main()
