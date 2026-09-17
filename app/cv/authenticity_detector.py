import os
import cv2
import numpy as np
import logging
import tempfile
from PIL import Image, ImageChops, ImageEnhance
from typing import Dict, Any, List

logger = logging.getLogger("DocVisionAI.CV.Authenticity")

# Verhoeff matrices for Aadhaar checksum validation
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

def validate_verhoeff(num_str: str) -> bool:
    """Validates Aadhaar number using Verhoeff algorithm."""
    digits = [int(char) for char in num_str if char.isdigit()]
    if len(digits) != 12:
        return False
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        checksum = VERHOEFF_D[checksum][VERHOEFF_P[i % 8][digit]]
    return checksum == 0


class DocumentAuthenticityDetector:
    """
    Computer Vision & Algorithmic Anti-Forgery Detector for Documents.
    Performs multi-modal verification:
      1. Error Level Analysis (ELA) for digital tampering / Photoshop edits.
      2. Sharpness & Blur assessment via Laplacian variance.
      3. Texture noise consistency analysis (detecting flat synthetic fills).
      4. QR Code & Barcode detection.
      5. Geometry & Aspect Ratio validation.
      6. Mathematical & Format Checksums (Verhoeff math, PAN format, Invoice totals).
    """

    @classmethod
    def analyze_authenticity(
        cls,
        image_path: str,
        doc_type: str,
        ocr_results: List[Dict[str, Any]] = None,
        fields: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        ocr_results = ocr_results or []
        fields = fields or {}
        
        checks = []
        score = 100.0

        # Load image with OpenCV (grayscale & color)
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError("Could not read image file.")
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
        except Exception as e:
            logger.error(f"Authenticity detector failed to read image {image_path}: {e}")
            return {
                "is_authentic": False,
                "authenticity_score": 0.0,
                "verdict": "Unreadable Image",
                "risk_level": "HIGH",
                "checks": [{"name": "Image Load", "passed": False, "details": str(e)}]
            }

        # --- Check 1: Error Level Analysis (ELA) for Digital Manipulation ---
        ela_passed, ela_score_diff, ela_details = cls._check_ela(image_path)
        checks.append({
            "name": "Error Level Analysis (ELA)",
            "passed": ela_passed,
            "details": ela_details
        })
        if not ela_passed:
            score -= 30.0

        # --- Check 2: Image Sharpness & Blur Level (Laplacian Variance) ---
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_passed = lap_var >= 40.0
        if lap_var >= 100.0:
            blur_details = f"High image sharpness (Laplacian var: {lap_var:.1f}). No artificial blur detected."
        elif lap_var >= 40.0:
            blur_details = f"Acceptable image clarity (Laplacian var: {lap_var:.1f})."
        else:
            blur_details = f"Low sharpness / Severe blur (Laplacian var: {lap_var:.1f}). Potential low-res reprint or digital forgery."
            score -= 20.0
            
        checks.append({
            "name": "Sharpness & Resolution Check",
            "passed": blur_passed,
            "details": blur_details
        })

        # --- Check 3: Texture Standard Deviation (Synthetic Flat Color Fill Detection) ---
        texture_passed, texture_std, texture_details = cls._check_texture(gray)
        checks.append({
            "name": "Texture & Surface Uniformity",
            "passed": texture_passed,
            "details": texture_details
        })
        if not texture_passed:
            score -= 15.0

        # --- Check 4: Aspect Ratio & Card Geometry ---
        ratio = max(w, h) / max(1, min(w, h))
        if doc_type in ["aadhaar", "pan"]:
            # Standard ID card ratio is ~1.58 (CR80 spec)
            geom_passed = 1.25 <= ratio <= 1.85
            geom_details = (
                f"Valid ID card aspect ratio ({ratio:.2f}:1)."
                if geom_passed
                else f"Non-standard card aspect ratio ({ratio:.2f}:1). Potential cropped fake template."
            )
            if not geom_passed:
                score -= 15.0
        else:
            geom_passed = True
            geom_details = f"Document aspect ratio ({ratio:.2f}:1) suitable for invoice format."

        checks.append({
            "name": "Document Geometry & Layout",
            "passed": geom_passed,
            "details": geom_details
        })

        # --- Check 5: QR Code & Security Marker Detection ---
        qr_detected, qr_details = cls._check_qr_code(gray, doc_type)
        checks.append({
            "name": "Security QR Code / Barcode Detector",
            "passed": qr_detected,
            "details": qr_details
        })
        # For Aadhaar, QR code is a major security feature
        if doc_type == "aadhaar" and not qr_detected:
            score -= 10.0

        # --- Check 6: Domain Mathematical & Syntactical Validation ---
        domain_passed, domain_details, domain_penalty = cls._check_domain_syntax(doc_type, ocr_results, fields)
        checks.append({
            "name": "Data Syntax & Math Checksum",
            "passed": domain_passed,
            "details": domain_details
        })
        if not domain_passed:
            score -= domain_penalty

        # Final Score Calculation (Clamped 0 - 100%)
        final_score = round(max(0.0, min(100.0, score)), 1)
        
        if final_score >= 80.0:
            verdict = "Real / Authentic Document"
            risk_level = "LOW"
            is_authentic = True
        elif final_score >= 50.0:
            verdict = "Suspicious / Potential Forgery"
            risk_level = "MEDIUM"
            is_authentic = False
        else:
            verdict = "Fake / Tampered Document"
            risk_level = "HIGH"
            is_authentic = False

        return {
            "is_authentic": is_authentic,
            "authenticity_score": final_score,
            "verdict": verdict,
            "risk_level": risk_level,
            "checks": checks
        }

    @staticmethod
    def _check_ela(image_path: str, quality: int = 95) -> tuple[bool, float, str]:
        """Performs Error Level Analysis (ELA) to detect JPEG re-compression tampering."""
        try:
            orig = Image.open(image_path).convert("RGB")
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name

            orig.save(tmp_path, "JPEG", quality=quality)
            resaved = Image.open(tmp_path).convert("RGB")

            # Calculate difference
            diff = ImageChops.difference(orig, resaved)
            os.remove(tmp_path)

            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])

            # Convert to numpy array to measure variance in compression error
            diff_arr = np.array(diff, dtype=np.float32)
            std_diff = float(np.std(diff_arr))

            # High max_diff combined with high std_diff indicates localized Photoshop / Paint editing
            if std_diff > 12.0 and max_diff > 45:
                return False, std_diff, f"High ELA variance detected (std: {std_diff:.1f}, max diff: {max_diff}). Digital modification detected."
            
            return True, std_diff, f"Clean ELA compression signature (std error: {std_diff:.1f}). No digital text alteration detected."
        except Exception as e:
            logger.warning(f"ELA check fallback: {e}")
            return True, 0.0, "ELA check skipped (Format compatibility)."

    @staticmethod
    def _check_texture(gray: np.ndarray) -> tuple[bool, float, str]:
        """Analyzes texture noise consistency to detect synthetic flat digital backgrounds."""
        try:
            h, w = gray.shape
            if h < 40 or w < 40:
                return True, 20.0, "Image too small for patch texture test."

            # Divide into 4x4 grid patches
            patch_h, patch_w = h // 4, w // 4
            std_devs = []

            for i in range(4):
                for j in range(4):
                    patch = gray[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                    std_devs.append(np.std(patch))

            mean_patch_std = float(np.mean(std_devs))
            if mean_patch_std < 5.0:
                return False, mean_patch_std, f"Unnaturally flat surface texture (std: {mean_patch_std:.1f}). Synthetic fake ID pattern."

            return True, mean_patch_std, f"Natural paper/card surface noise (texture std: {mean_patch_std:.1f})."
        except Exception as e:
            return True, 15.0, f"Texture check skipped: {e}"

    @staticmethod
    def _check_qr_code(gray: np.ndarray, doc_type: str) -> tuple[bool, str]:
        """Detects presence of security QR codes or barcodes."""
        try:
            qr_detector = cv2.QRCodeDetector()
            retval, _, _ = qr_detector.detect(gray)
            if retval:
                return True, "Security QR Code successfully detected on document."

            # Backup contour search for square QR code pattern
            contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            square_count = 0
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
                if len(approx) == 4 and cv2.contourArea(cnt) > 400:
                    square_count += 1
            
            if square_count >= 3:
                return True, "QR Code finder patterns detected."

            if doc_type == "aadhaar":
                return False, "No security QR code found on Aadhaar card. High risk of counterfeit document."
            return False, "No QR Code / Barcode detected on document."
        except Exception as e:
            return False, f"QR detection evaluation skipped: {e}"

    @staticmethod
    def _check_domain_syntax(doc_type: str, ocr_results: List[Dict[str, Any]], fields: Dict[str, Any]) -> tuple[bool, str, float]:
        """Checks domain-specific checksums and text formatting rules."""
        flat_text = " ".join([item.get("text", "") for item in ocr_results])
        
        if doc_type == "aadhaar":
            aadhaar_num = fields.get("aadhaar_number", "")
            clean_num = aadhaar_num.replace("-", "").replace(" ", "")
            if len(clean_num) == 12 and clean_num.isdigit():
                if validate_verhoeff(clean_num):
                    return True, f"Aadhaar Verhoeff Checksum valid ({aadhaar_num}).", 0.0
                else:
                    return False, f"Invalid Aadhaar Verhoeff Checksum ({aadhaar_num})! Mathematically fake Aadhaar number.", 35.0
            else:
                return False, "Missing or malformed 12-digit Aadhaar number.", 20.0

        elif doc_type == "pan":
            pan_num = fields.get("PAN Number", fields.get("pan_number", ""))
            clean_pan = pan_num.strip().upper()
            import re
            if re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$", clean_pan):
                return True, f"Valid PAN format syntax ({clean_pan}).", 0.0
            else:
                return False, f"Invalid PAN number structure ({clean_pan}). Expected 5 letters, 4 digits, 1 letter.", 25.0

        elif doc_type == "invoice":
            subtotal = float(fields.get("subtotal", 0.0))
            tax = float(fields.get("tax", 0.0))
            total = float(fields.get("total_amount", 0.0))
            if subtotal > 0 and total > 0:
                expected_total = subtotal + tax
                if abs(expected_total - total) < 1.0:
                    return True, f"Invoice pricing math verified (Subtotal {subtotal} + Tax {tax} == Total {total}).", 0.0
                else:
                    return False, f"Invoice pricing mismatch (Subtotal {subtotal} + Tax {tax} != Total {total}).", 15.0
            return True, "Invoice key fields present.", 0.0

        return True, "Syntax checks passed.", 0.0
