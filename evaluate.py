#!/usr/bin/env python
"""
DocVisionAI: Metric Evaluation Script.
Calculates Precision, Recall, F1, and Exact Match metrics comparing 
extracted JSON schemas with ground-truth labels. Saves training loss curves.
"""
import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.vlm_model import DocVisionVLM
from app.ocr.ocr_engine import OCREngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DocVisionAI.Evaluate")

# Optional matplotlib import
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("matplotlib not installed. Loss curves will be logged but not saved as images.")

def normalize_value(val: Any) -> str:
    """Normalizes field values for robust comparison."""
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        # Normalize pricing decimals to float representation
        return str(round(float(val), 2))
    return str(val).strip().lower().replace(" ", "")

def compute_json_metrics(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes Precision, Recall, F1 and Exact Match for structured document keys.
    """
    tp = 0
    fp = 0
    fn = 0
    
    gt_normalized = {str(k).lower(): normalize_value(v) for k, v in ground_truth.items()}
    pred_normalized = {str(k).lower(): normalize_value(v) for k, v in prediction.items()}
    
    # Exact Match
    exact_match = 1.0 if gt_normalized == pred_normalized else 0.0
    
    # True Positives & False Positives
    for k, v in pred_normalized.items():
        if k in gt_normalized:
            if v == gt_normalized[k]:
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
            
    # False Negatives
    for k in gt_normalized.keys():
        if k not in pred_normalized:
            fn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match
    }

def plot_loss_curves(summary_path: str, output_image_path: str):
    """Plots and saves loss curves using training metadata."""
    if not os.path.exists(summary_path):
        logger.warning(f"Training summary file not found at {summary_path}. Skipping loss curve plotting.")
        return

    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
            
        history = summary.get("loss_history", [])
        if not history:
            logger.warning("No loss history found in training summary. Cannot plot curves.")
            return

        steps = [h.get("step") or h.get("epoch") for h in history]
        losses = [h.get("loss") for h in history]
        
        logger.info(f"Loss history: {history}")
        
        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=(8, 5))
            plt.plot(steps, losses, marker="o", color="#6366f1", linewidth=2, label="Train Loss")
            plt.title("DocVisionAI - Training Cross-Entropy Loss Curve")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_image_path, dpi=150)
            plt.close()
            logger.info(f"Saved loss curve image to: {output_image_path}")
        else:
            logger.info("ASCII Loss Curve View:")
            for h in history:
                logger.info(f"  Step {h.get('step') or h.get('epoch')}: {'*' * int(max(1, h.get('loss') * 10))} ({h.get('loss')})")
    except Exception as e:
        logger.error(f"Error plotting loss curves: {e}")

def main():
    parser = argparse.ArgumentParser(description="DocVisionAI Evaluation CLI")
    parser.add_argument("--eval_data", type=str, default="datasets/splits/eval.json", help="Path to evaluation JSON split")
    parser.add_argument("--summary_path", type=str, default="lora_adapters/training_summary.json", help="Path to training summary file")
    parser.add_argument("--loss_curve_path", type=str, default="evaluation_loss_curve.png", help="Where to save the loss curve plot")
    
    args = parser.parse_args()
    
    # Plot loss curve from training metadata
    plot_loss_curves(args.summary_path, args.loss_curve_path)
    
    if not os.path.exists(args.eval_data):
        logger.error(f"Evaluation file not found: {args.eval_data}. Please specify a valid file path.")
        sys.exit(1)
        
    logger.info(f"Loading evaluation dataset: {args.eval_data}")
    with open(args.eval_data, "r") as f:
        eval_samples = json.load(f)
        
    logger.info(f"Initialized {len(eval_samples)} evaluation samples. Executing batch validation...")
    
    # Init inference engines
    vlm = DocVisionVLM()
    ocr = OCREngine()
    
    total_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "exact_match": 0.0
    }
    
    count = 0
    for sample in eval_samples:
        image_path = sample.get("image")
        conversations = sample.get("conversations", [])
        doc_type = sample.get("doc_type", "invoice")
        
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Image not found at {image_path}, skipping sample.")
            continue
            
        # Ground truth is assistant's response content
        try:
            gt_text = conversations[1]["content"][0]["text"]
            ground_truth = json.loads(gt_text)
        except Exception as e:
            logger.error(f"Failed to parse ground truth annotation for sample {sample.get('id')}: {e}")
            continue
            
        # Run OCR
        ocr_results, _ = ocr.extract_text(image_path, doc_type)
        
        # Run VLM
        result = vlm.process_document(image_path, ocr_results, doc_type)
        prediction = result.get("fields", {})
        
        # Calculate scores
        metrics = compute_json_metrics(prediction, ground_truth)
        
        for k in total_metrics.keys():
            total_metrics[k] += metrics[k]
            
        count += 1
        
    if count == 0:
        logger.error("No valid evaluation samples were successfully evaluated.")
        sys.exit(1)
        
    # Average metrics
    avg_metrics = {k: round(v / count, 4) for k, v in total_metrics.items()}
    
    logger.info("==========================================")
    logger.info("METRIC EVALUATION SUMMARY")
    logger.info("==========================================")
    logger.info(f"Evaluated Samples: {count}")
    logger.info(f"Precision: {avg_metrics['precision']:.4f}")
    logger.info(f"Recall:    {avg_metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {avg_metrics['f1']:.4f}")
    logger.info(f"Exact Match: {avg_metrics['exact_match']:.4f}")
    logger.info("==========================================")
    
    # Save evaluation summary JSON
    eval_summary_path = os.path.join(os.path.dirname(args.eval_data), "eval_results.json")
    with open(eval_summary_path, "w") as f:
        json.dump({
            "metrics": avg_metrics,
            "samples_count": count,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    logger.info(f"Saved evaluation metrics to: {eval_summary_path}")

if __name__ == "__main__":
    import time
    main()
