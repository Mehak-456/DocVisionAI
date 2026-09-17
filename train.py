#!/usr/bin/env python
"""
DocVisionAI: Qwen2.5-VL LoRA Fine-Tuning CLI.
Triggers training of the vision-language model with custom parameters.
"""
import os
import sys
import argparse
import logging

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.training.train_engine import DocVisionTrainer
from app.datasets.dataset_loader import DocumentDatasetLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DocVisionAI.TrainCLI")

def main():
    parser = argparse.ArgumentParser(description="DocVisionAI Qwen2.5-VL LoRA Fine-Tuning CLI")
    
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Train batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to save adapters")
    parser.add_argument("--train_data", type=str, default=None, help="Path to train.json file")
    parser.add_argument("--eval_data", type=str, default=None, help="Path to eval.json file")
    parser.add_argument("--data_root", type=str, default="datasets", help="Root folder to search for images/annotations")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Split ratio if data paths aren't specified")

    args = parser.parse_args()

    train_path = args.train_data
    eval_path = args.eval_data

    # If dataset files aren't explicitly provided, scan folders and generate splits
    if not train_path or not eval_path:
        logger.info(f"Scanning for document samples in root '{args.data_root}'...")
        loader = DocumentDatasetLoader(data_root=args.data_root)
        train_path, eval_path = loader.create_and_save_splits(
            train_ratio=args.split_ratio, 
            output_dir=os.path.join(args.data_root, "splits")
        )

    # Initialize trainer
    trainer = DocVisionTrainer(
        train_data_path=train_path,
        eval_data_path=eval_path,
        output_dir=args.output_dir
    )

    logger.info("Initializing LoRA training execution...")
    result = trainer.run_training(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum
    )

    if result.get("success"):
        logger.info("==========================================")
        logger.info("TRAINING SUCCESSFULLY FINISHED!")
        logger.info(f"Mode: {result.get('mode')}")
        logger.info(f"Saved Adapters: {result.get('adapter_path')}")
        logger.info(f"Final training loss: {result.get('training_loss'):.4f}")
        logger.info("==========================================")
    else:
        logger.error("Training execution failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
