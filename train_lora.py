"""
DocVisionAI: Qwen2.5-VL Fine-Tuning Script with PEFT / LoRA
Designed for PyTorch, Hugging Face Transformers, and PEFT library.
This script demonstrates how to configure, set up, and train Qwen2.5-VL with LoRA adapters 
to perform structured information extraction on custom invoices and ID cards.
"""

import os
import torch
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
from datasets import load_dataset
from PIL import Image

# Import PEFT and Transformers
try:
    import transformers
    from transformers import (
        AutoProcessor,
        Trainer,
        TrainingArguments,
        Qwen2_5_VLForConditionalGeneration
    )
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    print("Warning: PEFT and Transformers libraries are required to run this training script in production.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocVisionAI.Train")

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    use_flash_attn: bool = field(default=True)

@dataclass
class DataArguments:
    train_data_path: str = field(default="./dataset/train.json", metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default="./dataset/eval.json", metadata={"help": "Path to the evaluation data."})
    image_folder: str = field(default="./dataset/images", metadata={"help": "Directory containing images."})

@dataclass
class LoraArguments:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # For Vision-Language models like Qwen2.5-VL, we target attention layers in both vision and text encoders.
    # Specifying these targets ensures LoRA fine-tunes the language projections and the visual visual attention layers.
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", 
            "v_proj", 
            "k_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj", 
            "down_proj"
        ]
    )

def configure_lora(model, lora_args: LoraArguments):
    """
    Applies LoRA adapter layers to target modules of Qwen2.5-VL.
    This cuts trainable parameters to < 1% of the model while retaining high performance.
    """
    peft_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    logger.info("Wrapping model with PEFT/LoRA...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

class CollatorForMultimodalDataset:
    """
    Data collator that formats conversations and loads images for Qwen2.5-VL processor.
    Handles visual token padding and alignment.
    """
    def __init__(self, processor: AutoProcessor, image_folder: str):
        self.processor = processor
        self.image_folder = image_folder

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_text = []
        batch_images = []

        for inst in instances:
            # Build conversation messages list
            messages = inst["conversations"]
            image_file = inst.get("image")
            
            # Load corresponding PIL Image
            if image_file:
                image_path = os.path.join(self.image_folder, image_file)
                try:
                    img = Image.open(image_path).convert("RGB")
                    batch_images.append(img)
                except Exception as e:
                    logger.error(f"Failed to load image {image_path}: {e}")
                    # Create a blank fallback image
                    batch_images.append(Image.new("RGB", (224, 224), color="white"))
            
            # Format using Qwen2.5-VL chat template
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            batch_text.append(text)

        # Process text and images jointly
        inputs = self.processor(
            text=batch_text,
            images=batch_images if batch_images else None,
            padding=True,
            return_tensors="pt"
        )

        # Labels are target shifted input ids (typical causal language model setup)
        labels = inputs.input_ids.clone()
        # Set padding tokens and prompt tokens labels to -100 to ignore in loss calculation
        # (This avoids calculating loss on the user prompt/image tokens and only optimizes on the JSON output)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # Mask out prompts (everything before assistant turn)
        # Note: A real implementation would parse conversation segments to mask user turns.
        inputs["labels"] = labels

        return inputs

def train():
    # Parse arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f"Loading processor and base model: {model_args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    
    # Load model in bfloat16/float16
    device_map = "auto"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        use_flash_attention_2=model_args.use_flash_attn
    )

    # Enable gradient checkpointing to save VRAM (essential for 7B models on 24GB GPUs)
    model.gradient_checkpointing_enable()

    # Wrap model with LoRA adapters
    model = configure_lora(model, lora_args)

    # Load custom structured dataset (JSON)
    # Schema expected: 
    # [
    #   {
    #     "id": "invoice_001",
    #     "image": "invoice_001.jpg",
    #     "conversations": [
    #       {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Extract structured details from invoice."}]},
    #       {"role": "assistant", "content": [{"type": "text", "text": "{\"invoice_number\": \"INV-100\"..."}]}
    #     ]
    #   }
    # ]
    logger.info(f"Loading train dataset: {data_args.train_data_path}")
    dataset = load_dataset("json", data_files={"train": data_args.train_data_path})
    train_dataset = dataset["train"]

    # Setup data collator
    data_collator = CollatorForMultimodalDataset(processor, data_args.image_folder)

    # Initialize Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )

    # Launch Training
    logger.info("Launching Trainer training loop...")
    trainer.train()
    
    # Save the custom LoRA adapters
    logger.info(f"Saving LoRA adapters to: {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info("Training complete!")

if __name__ == "__main__":
    # If executed, this script demonstrates execution path. 
    # Under typical development environments, it acts as a reference structure.
    print("DocVisionAI - Qwen2.5-VL LoRA Fine-Tuning Module initialized.")
