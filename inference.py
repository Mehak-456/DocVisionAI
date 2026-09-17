#!/usr/bin/env python
"""
DocVisionAI: Programmatic SDK Inference Interface.
Exposes a unified pipeline class to process documents in external codebases.
"""
import os
import sys
import logging
from typing import Dict, Any, Tuple

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.preprocessing.image_processor import ImagePreprocessor
from app.ocr.ocr_engine import OCREngine
from app.models.vlm_model import DocVisionVLM
from app.configs.config import settings

logger = logging.getLogger("DocVisionAI.SDK")

class DocVisionPipeline:
    def __init__(self, use_simulator: bool = None, quantization: str = None, ocr_gpu: bool = False):
        """
        Initializes the DocVisionAI end-to-end extraction pipeline.
        
        Args:
            use_simulator: Overrides the global USE_SIMULATOR environment variable.
            quantization: Quantization type for VLM: None, '8bit', or '4bit'.
            ocr_gpu: Enable CUDA GPU acceleration for EasyOCR.
        """
        # Load local configurations
        self.preprocessor = ImagePreprocessor()
        self.ocr = OCREngine(gpu=ocr_gpu)
        self.vlm = DocVisionVLM(use_simulator=use_simulator, quantization=quantization)
        
        logger.info(f"DocVisionPipeline initialized. Simulator mode: {self.vlm.use_simulator}")

    def run(self, file_path: str, doc_type: str = "auto", output_annotated_path: str = None) -> Dict[str, Any]:
        """
        Executes the full document understanding pipeline on an image or PDF.
        
        Steps:
        1. Converts PDF to image if required.
        2. Applies deskewing, edge-preserving denoising, resizing, and CLAHE.
        3. Extracts text bounding boxes using EasyOCR.
        4. (Optional) Generates annotated visual outline image.
        5. Performs fine-tuned Qwen2.5-VL LoRA query extraction.
        
        Args:
            file_path: Path to the input image (PNG, JPG, WEBP) or PDF.
            doc_type: Target schema: 'auto', 'invoice', 'aadhaar', or 'pan'.
            output_annotated_path: If specified, saves the OCR bounded image to this path.
            
        Returns:
            Dict[str, Any]: structured output including fields, document type, and confidence.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found at: {file_path}")

        temp_image_path = file_path
        is_pdf = file_path.lower().endswith(".pdf")
        
        # 1. PDF Conversion
        if is_pdf:
            import time
            temp_dir = os.path.dirname(os.path.abspath(file_path))
            temp_name = f"temp_pdf_page_{int(time.time() * 1000)}.jpg"
            converted_img_path = os.path.join(temp_dir, temp_name)
            temp_image_path = self.ocr.convert_pdf_to_image(file_path, converted_img_path)

        # 2. Image Preprocessing
        preprocessed_path = temp_image_path + ".preprocessed.jpg"
        try:
            self.preprocessor.preprocess_image(temp_image_path, preprocessed_path)
            work_image_path = preprocessed_path
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}. Falling back to source image.")
            work_image_path = temp_image_path

        # 3. Extract Text via OCR
        ocr_results, detected_doc_type = self.ocr.extract_text(work_image_path, doc_type)

        # 4. Optional: Render bounding boxes
        if output_annotated_path:
            self.ocr.draw_bboxes(work_image_path, ocr_results, output_annotated_path)
            logger.info(f"Saved OCR-visualizations to: {output_annotated_path}")

        # 5. Extract Structured fields via VLM
        logger.info(f"Querying VLM model for document type: {detected_doc_type}")
        vlm_result = self.vlm.process_document(work_image_path, ocr_results, detected_doc_type)

        # 6. Cleanup temporary files
        if os.path.exists(preprocessed_path):
            try:
                os.remove(preprocessed_path)
            except OSError:
                pass
                
        if is_pdf and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except OSError:
                pass

        return {
            "success": True,
            "document_type": detected_doc_type,
            "fields": vlm_result.get("fields", {}),
            "confidence": vlm_result.get("confidence", 0.0),
            "latency_sec": vlm_result.get("inference_latency_sec", 0.0),
            "ocr_boxes_found": len(ocr_results),
            "vlm_mode": vlm_result.get("model_mode", "Simulated")
        }

    def export_onnx(self, output_path: str) -> bool:
        """
        Triggers ONNX tracing and exports the loaded model.
        """
        return self.vlm.export_to_onnx(output_path)
