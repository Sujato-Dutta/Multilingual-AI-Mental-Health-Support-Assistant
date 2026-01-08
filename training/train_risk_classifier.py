"""
Risk Classifier Training Script

Fine-tunes DistilRoBERTa for 3-class risk classification using LoRA.
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import Dict
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split

from configs.model_config import CONFIG
from utils.seed import set_all_seeds
from utils.logging_utils import get_logger

logger = get_logger(__name__, log_dir="logs")


def load_dataset(data_path: str) -> tuple:
    """Load and prepare the risk classification dataset."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    
    return texts, labels


def compute_class_weights(labels: list) -> torch.Tensor:
    """Compute class weights for imbalanced data."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(3):
        weight = total / (3 * counts[i]) if counts[i] > 0 else 1.0
        weights.append(weight)
    # Increase weight for HIGH risk class
    weights[2] *= 1.5
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(eval_pred) -> Dict:
    """Compute evaluation metrics with focus on HIGH risk recall."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Per-class metrics
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    
    # HIGH class recall (most important)
    high_recall = recall_score(labels, predictions, labels=[2], average='macro', zero_division=0)
    
    # Macro F1
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        'accuracy': (predictions == labels).mean(),
        'macro_f1': macro_f1,
        'high_recall': high_recall,
        'low_f1': report['0']['f1-score'] if '0' in report else 0,
        'medium_f1': report['1']['f1-score'] if '1' in report else 0,
        'high_f1': report['2']['f1-score'] if '2' in report else 0,
    }


def main():
    """Main training function."""
    # Set seeds
    set_all_seeds(CONFIG.inference.random_seed)
    
    config = CONFIG.risk_classifier
    
    logger.info("Starting risk classifier training")
    logger.info(f"Base model: {config.base_model}")
    
    # Load dataset
    data_path = os.path.join("data", "risk_dataset.json")
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found: {data_path}")
        return
    
    texts, labels = load_dataset(data_path)
    logger.info(f"Loaded {len(texts)} samples")
    
    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
    
    # Import required libraries
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=3,
        id2label={0: "LOW", 1: "MEDIUM", 2: "HIGH"},
        label2id={"LOW": 0, "MEDIUM": 1, "HIGH": 2},
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare datasets
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding=False
        )
    
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    output_dir = config.adapter_path
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="high_recall",
        greater_is_better=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        report_to="none",
        fp16=False,
        dataloader_num_workers=0,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Custom trainer with class weights
    class WeightedTrainer(Trainer):
        def __init__(self, class_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss
    
    # Compute class weights
    class_weights = compute_class_weights(train_labels)
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Train
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate
    logger.info("Evaluating...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    # Print confusion matrix
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    cm = confusion_matrix(val_labels, preds)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")
    print(f"HIGH risk recall: {eval_results.get('eval_high_recall', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
