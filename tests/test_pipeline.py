import os
import sys
import unittest
import numpy as np
from PIL import Image

# Ensure root folder is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.preprocessing.image_processor import ImagePreprocessor
from app.ocr.ocr_engine import OCREngine
from app.models.vlm_model import DocVisionVLM
from inference import DocVisionPipeline

class TestDocVisionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a dummy image for testing
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.dummy_img_path = os.path.join(cls.test_dir, "test_invoice.jpg")
        
        # Draw a white image with some simulated text blocks
        img = Image.new("RGB", (800, 1000), color="white")
        img.save(cls.dummy_img_path)

    @classmethod
    def tearDownClass(cls):
        # Cleanup dummy image
        if os.path.exists(cls.dummy_img_path):
            os.remove(cls.dummy_img_path)

    def test_image_preprocessor(self):
        """Tests image preprocessing operations."""
        img = ImagePreprocessor.read_image_cv2(self.dummy_img_path)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[0], 1000)
        self.assertEqual(img.shape[1], 800)
        
        # Test Resizing
        resized = ImagePreprocessor.resize(img, target_width=500)
        self.assertEqual(resized.shape[1], 500)
        
        # Test Denoising
        denoised = ImagePreprocessor.denoise(resized)
        self.assertEqual(denoised.shape, resized.shape)

    def test_ocr_engine_simulation(self):
        """Tests OCR text box detection and schema recognition."""
        ocr = OCREngine()
        # Test auto-detection mapping
        results, doc_type = ocr.extract_text(self.dummy_img_path, doc_type="invoice")
        self.assertEqual(doc_type, "invoice")
        self.assertTrue(len(results) > 0)
        
        # Bbox shape validation: 4 points of 2D coordinates
        first_box = results[0]["box"]
        self.assertEqual(len(first_box), 4)
        self.assertEqual(len(first_box[0]), 2)

    def test_vlm_model_extraction(self):
        """Tests information extraction on VLM class."""
        vlm = DocVisionVLM(use_simulator=True)
        ocr = OCREngine()
        ocr_results, _ = ocr.extract_text(self.dummy_img_path, doc_type="invoice")
        
        # Test invoice extraction schema
        result = vlm.process_document(self.dummy_img_path, ocr_results, doc_type="invoice")
        self.assertEqual(result["document_type"], "invoice")
        self.assertIn("invoice_number", result["fields"])
        self.assertIn("vendor_name", result["fields"])
        self.assertTrue(result["confidence"] > 0.0)

    def test_end_to_end_pipeline(self):
        """Tests the unified SDK pipeline runner."""
        pipeline = DocVisionPipeline(use_simulator=True)
        out_annotated = self.dummy_img_path + ".annotated.jpg"
        
        result = pipeline.run(
            file_path=self.dummy_img_path, 
            doc_type="invoice", 
            output_annotated_path=out_annotated
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["document_type"], "invoice")
        self.assertIn("invoice_number", result["fields"])
        
        # Verify annotated file was generated and clean it up
        self.assertTrue(os.path.exists(out_annotated))
        if os.path.exists(out_annotated):
            os.remove(out_annotated)

if __name__ == "__main__":
    unittest.main()
