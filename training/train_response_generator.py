"""
Response Generator Training Script

Fine-tunes Qwen2.5-0.5B-Instruct using QLoRA for instruction-tuned responses.
"""

import json
import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import Dict

from configs.model_config import CONFIG
from utils.seed import set_all_seeds
from utils.logging_utils import get_logger

logger = get_logger(__name__, log_dir="logs")


def load_dataset(data_path: str) -> list:
    """Load instruction-tuning dataset."""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_conversation(sample: Dict, tokenizer) -> str:
    """Format a sample as a conversation for training."""
    messages = [
        {"role": "system", "content": sample["system"]},
        {"role": "user", "content": sample["user"]},
        {"role": "assistant", "content": sample["assistant"]}
    ]
    
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    
    return f"System: {sample['system']}\nUser: {sample['user']}\nAssistant: {sample['assistant']}"


def main():
    """Main training function."""
    set_all_seeds(CONFIG.inference.random_seed)
    
    config = CONFIG.response_generator
    qlora_config = CONFIG.qlora
    
    logger.info("Starting response generator training")
    logger.info(f"Base model: {config.base_model}")
    
    # Load dataset
    data_path = os.path.join("data", "instruction_dataset.json")
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found: {data_path}")
        return
    
    data = load_dataset(data_path)
    logger.info(f"Loaded {len(data)} samples")
    
    # Split
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Import libraries
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model (CPU-only, no quantization)
    logger.info("Loading model for CPU training (no quantization)")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    # Enable gradients for all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=qlora_config.lora_r,
        lora_alpha=qlora_config.lora_alpha,
        lora_dropout=qlora_config.lora_dropout,
        target_modules=qlora_config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    
    # Ensure LoRA parameters have gradients enabled
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    
    model.print_trainable_parameters()
    
    # Prepare datasets
    def preprocess_function(examples):
        texts = [format_conversation({"system": s, "user": u, "assistant": a}, tokenizer) 
                 for s, u, a in zip(examples['system'], examples['user'], examples['assistant'])]
        tokenized = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
        # Deep copy input_ids for labels
        tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]
        return tokenized
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)
    
    # Training arguments
    output_dir = config.instruction_adapter_path
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Reduced for faster training
        per_device_train_batch_size=1,  # Small batch for CPU
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Accumulate gradients
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        report_to="none",
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    print("\nInstruction tuning complete!")
    print(f"Adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
