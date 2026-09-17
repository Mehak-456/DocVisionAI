import cv2
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger("DocVisionAI.Preprocessing")

class ImagePreprocessor:
    @staticmethod
    def read_image_cv2(image_path: str) -> np.ndarray:
        """Reads an image path into a numpy array (BGR format)."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image at {image_path}")
        return img

    @staticmethod
    def save_image_cv2(image_path: str, img: np.ndarray) -> None:
        """Saves a BGR numpy array image to path."""
        cv2.imwrite(image_path, img)

    @staticmethod
    def deskew(img: np.ndarray) -> np.ndarray:
        """
        Detects document skew angle and rotates the image to align it.
        Uses thresholding, contour finding, and minAreaRect to calculate skew.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Threshold to get text areas
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find coordinates of all thresholded pixels
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) == 0:
                logger.warning("No text detected for deskewing, skipping.")
                return img
                
            # Get bounding box of all text pixels
            angle = cv2.minAreaRect(coords)[-1]
            
            # minAreaRect returns angle in range [-90, 0)
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Ignore tiny skews to avoid image degradation
            if abs(angle) < 0.5 or abs(angle) > 45:
                return img
                
            logger.info(f"Detected skew angle: {angle:.2f} degrees. Rotating...")
            
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception as e:
            logger.error(f"Error during deskewing: {e}. Returning original.")
            return img

    @staticmethod
    def denoise(img: np.ndarray) -> np.ndarray:
        """
        Applies edge-preserving denoising.
        Uses cv2.bilateralFilter to smooth background noise while keeping text edges crisp.
        """
        try:
            # Bilateral filter params: diameter=9, sigmaColor=75, sigmaSpace=75
            denoised = cv2.bilateralFilter(img, 9, 75, 75)
            return denoised
        except Exception as e:
            logger.error(f"Error during denoising: {e}. Returning original.")
            return img

    @staticmethod
    def resize(img: np.ndarray, target_width: int = 1200) -> np.ndarray:
        """
        Resizes the image to a target width while maintaining the aspect ratio.
        """
        try:
            h, w = img.shape[:2]
            if w <= target_width:
                return img
                
            scale = target_width / w
            target_height = int(h * scale)
            resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {w}x{h} to {target_width}x{target_height}")
            return resized
        except Exception as e:
            logger.error(f"Error during resizing: {e}. Returning original.")
            return img

    @staticmethod
    def enhance_contrast(img: np.ndarray) -> np.ndarray:
        """
        Enhances image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Applies CLAHE on the L-channel of the LAB color space to preserve natural colors.
        """
        try:
            # Convert BGR to LAB
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge back and convert to BGR
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return enhanced
        except Exception as e:
            logger.error(f"Error during contrast enhancement: {e}. Returning original.")
            return img

    @classmethod
    def preprocess_image(cls, input_path: str, output_path: str) -> None:
        """
        Runs the full preprocessing pipeline:
        Read -> Resize -> Deskew -> Denoise -> Enhance Contrast -> Save.
        """
        logger.info(f"Preprocessing image: {input_path}")
        img = cls.read_image_cv2(input_path)
        
        # 1. Resize first to make operations faster
        img = cls.resize(img, target_width=1200)
        
        # 2. Deskew to align text
        img = cls.deskew(img)
        
        # 3. Denoise to remove artifacts
        img = cls.denoise(img)
        
        # 4. Enhance contrast for OCR readability
        img = cls.enhance_contrast(img)
        
        cls.save_image_cv2(output_path, img)
        logger.info(f"Preprocessed image saved to: {output_path}")
