"""
Model Configuration Module

Contains all model paths, hyperparameters, QLoRA settings, and generation parameters
for the AI Mental Health Support Assistant.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QLoRAConfig:
    """QLoRA configuration for parameter-efficient fine-tuning."""
    
    # Quantization settings
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    
    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class RiskClassifierConfig:
    """Configuration for risk classification model."""
    
    base_model: str = "distilroberta-base"
    adapter_path: str = "adapters/risk_classifier"
    num_labels: int = 3
    label_map: dict = field(default_factory=lambda: {
        0: "LOW",
        1: "MEDIUM", 
        2: "HIGH"
    })
    
    # LoRA settings for classifier
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "query", "key", "value"
    ])
    
    # Training settings
    learning_rate: float = 2e-4
    num_epochs: int = 5
    batch_size: int = 8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1


@dataclass
class ResponseGeneratorConfig:
    """Configuration for response generation model."""
    
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    instruction_adapter_path: str = "adapters/instruction_adapter"
    safety_adapter_path: str = "adapters/safety_adapter"
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    
    # Deterministic settings for high-risk
    deterministic_temperature: float = 0.0
    deterministic_do_sample: bool = False
    
    # Training settings
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048


@dataclass
class WhisperConfig:
    """Configuration for Whisper speech-to-text."""
    
    model_name: str = "openai/whisper-small"
    device: str = "cpu"
    language: Optional[str] = None
    task: str = "transcribe"


@dataclass
class TranslationConfig:
    """Configuration for translation pipeline."""
    
    # Supported language codes for argos-translate
    supported_languages: List[str] = field(default_factory=lambda: [
        "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko",
        "ar", "hi", "nl", "pl", "tr", "vi", "th", "id", "sv", "da"
    ])
    fallback_language: str = "en"
    max_text_length: int = 5000


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    
    # Memory settings
    max_conversation_turns: int = 5
    
    # Deterministic behavior
    random_seed: int = 42
    
    # Logging
    log_model_versions: bool = True
    log_adapter_hashes: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Environment settings
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # Model configurations
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)
    risk_classifier: RiskClassifierConfig = field(default_factory=RiskClassifierConfig)
    response_generator: ResponseGeneratorConfig = field(default_factory=ResponseGeneratorConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Paths
    data_dir: str = "data"
    adapters_dir: str = "adapters"
    logs_dir: str = "logs"


def get_config() -> AppConfig:
    """Get the application configuration."""
    return AppConfig()


# Global config instance
CONFIG = get_config()
