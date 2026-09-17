import os
import json
import logging
import glob
import random
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("DocVisionAI.DatasetLoader")

class DocumentDatasetLoader:
    def __init__(self, data_root: str = "datasets"):
        """
        data_root: Path containing invoice/, aadhaar/, and pan/ folders.
        """
        self.data_root = data_root
        self.supported_types = ["invoice", "aadhaar", "pan"]

    def find_samples(self) -> List[Dict[str, Any]]:
        """
        Scans folders for image and annotation pairs.
        Returns:
            list of dict: [{"image_path": "...", "annotation_path": "...", "doc_type": "..."}]
        """
        samples = []
        for doc_type in self.supported_types:
            folder_path = os.path.join(self.data_root, doc_type)
            if not os.path.exists(folder_path):
                logger.warning(f"Dataset subdirectory not found: {folder_path}")
                continue

            # Find all image files
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(folder_path, ext)))

            for img_path in image_files:
                # Expect corresponding JSON with same basename
                base_name, _ = os.path.splitext(img_path)
                ann_path = base_name + ".json"
                
                if os.path.exists(ann_path):
                    samples.append({
                        "image_path": img_path,
                        "annotation_path": ann_path,
                        "doc_type": doc_type
                    })
                else:
                    logger.debug(f"Annotation file not found for image: {img_path}")

        logger.info(f"Discovered {len(samples)} valid image-annotation pairs in root '{self.data_root}'.")
        return samples

    def format_for_qwen_vl(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Formats list of samples to the official Qwen2.5-VL Multi-modal conversational schema.
        """
        formatted = []
        for idx, sample in enumerate(samples):
            try:
                # Read target annotation JSON
                with open(sample["annotation_path"], "r", encoding="utf-8") as f:
                    ann_data = json.load(f)

                # Prompt instructions depending on document type
                doc_type = sample["doc_type"]
                if doc_type == "invoice":
                    prompt_text = "Extract structured details from this invoice: invoice_number, invoice_date, vendor_name, GSTIN, buyer_name, total_amount, subtotal, tax, currency."
                elif doc_type == "aadhaar":
                    prompt_text = "Extract details from this Aadhaar card: name, aadhaar_number, DOB, gender, address."
                elif doc_type == "pan":
                    prompt_text = "Extract details from this PAN card: PAN Number, Name, Father's Name, DOB."
                else:
                    prompt_text = "Extract structured details from this document."

                # Qwen2.5-VL conversational structure
                # User block includes image token + instruction text
                # Assistant block contains stringified JSON of fields
                formatted_sample = {
                    "id": f"doc_{idx}",
                    "image": os.path.abspath(sample["image_path"]),
                    "doc_type": doc_type,
                    "conversations": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt_text}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": json.dumps(ann_data)}
                            ]
                        }
                    ]
                }
                formatted.append(formatted_sample)
            except Exception as e:
                logger.error(f"Error parsing sample {sample['image_path']}: {e}")

        return formatted

    def create_and_save_splits(self, train_ratio: float = 0.8, output_dir: str = "dataset_splits") -> Tuple[str, str]:
        """
        Processes dataset directories, splits into train/eval subsets, and saves as JSON files.
        Returns paths to train and eval JSON logs.
        """
        samples = self.find_samples()
        if not samples:
            # Generate mock dataset files if no user data exists, keeping execution out-of-the-box ready
            logger.warning("No dataset files found! Generating mock dataset splits for pipeline testing.")
            return self._generate_mock_dataset_splits(output_dir)

        formatted_samples = self.format_for_qwen_vl(samples)
        
        # Train/Test Split
        if len(formatted_samples) > 1:
            shuffled = list(formatted_samples)
            random.seed(42)
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * train_ratio)
            train_data = shuffled[:split_idx]
            eval_data = shuffled[split_idx:]
            if not train_data:
                train_data = shuffled
                eval_data = shuffled
        else:
            train_data = formatted_samples
            eval_data = formatted_samples

        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train.json")
        eval_path = os.path.join(output_dir, "eval.json")

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, indent=2)

        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2)

        logger.info(f"Saved {len(train_data)} train samples to {train_path}")
        logger.info(f"Saved {len(eval_data)} eval samples to {eval_path}")

        return train_path, eval_path

    def _generate_mock_dataset_splits(self, output_dir: str) -> Tuple[str, str]:
        """
        Generates structured mock datasets for invoices, aadhaar, and pan.
        Ensures training scripts run end-to-end even in a clean checkout.
        """
        # Create directories for physical images if needed
        mock_img_dir = os.path.join(self.data_root, "mock_images")
        os.makedirs(mock_img_dir, exist_ok=True)
        
        # Create a tiny 100x100 white placeholder image
        from PIL import Image
        mock_image_path = os.path.join(mock_img_dir, "mock_doc.png")
        img = Image.new("RGB", (100, 100), color="white")
        img.save(mock_image_path)
        
        mock_samples = []
        
        # 1. Invoice Sample
        mock_samples.append({
            "id": "mock_invoice_1",
            "image": os.path.abspath(mock_image_path),
            "doc_type": "invoice",
            "conversations": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Extract structured details from this invoice: invoice_number, invoice_date, vendor_name, GSTIN, buyer_name, total_amount, subtotal, tax, currency."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": json.dumps({
                            "invoice_number": "INV-2026-999",
                            "invoice_date": "2026-07-28",
                            "vendor_name": "CloudTech Solutions",
                            "GSTIN": "27AAAAA1111A1Z1",
                            "buyer_name": "Vivek Chauhan",
                            "total_amount": 1628.40,
                            "subtotal": 1380.00,
                            "tax": 248.40,
                            "currency": "USD"
                        })}
                    ]
                }
            ]
        })

        # 2. Aadhaar Sample
        mock_samples.append({
            "id": "mock_aadhaar_1",
            "image": os.path.abspath(mock_image_path),
            "doc_type": "aadhaar",
            "conversations": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Extract details from this Aadhaar card: name, aadhaar_number, DOB, gender, address."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": json.dumps({
                            "name": "Vivek Chauhan",
                            "aadhaar_number": "1234-5678-9012",
                            "DOB": "15/08/1998",
                            "gender": "Male",
                            "address": "Delhi, India"
                        })}
                    ]
                }
            ]
        })

        # 3. PAN Sample
        mock_samples.append({
            "id": "mock_pan_1",
            "image": os.path.abspath(mock_image_path),
            "doc_type": "pan",
            "conversations": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Extract details from this PAN card: PAN Number, Name, Father's Name, DOB."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": json.dumps({
                            "PAN Number": "ABCDE1234F",
                            "Name": "VIVEK CHAUHAN",
                            "Father's Name": "R. S. CHAUHAN",
                            "DOB": "15/08/1998"
                        })}
                    ]
                }
            ]
        })

        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train.json")
        eval_path = os.path.join(output_dir, "eval.json")

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(mock_samples, f, indent=2)

        # Use same sample for eval
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(mock_samples, f, indent=2)

        logger.info(f"Generated {len(mock_samples)} mock train samples in {train_path}")
        return train_path, eval_path
