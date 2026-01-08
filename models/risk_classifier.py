"""
Risk Classifier Module

Provides risk classification for user inputs using a fine-tuned
DistilRoBERTa model with LoRA adapters.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from configs.model_config import CONFIG
from utils.logging_utils import get_logger
from utils.seed import set_all_seeds

logger = get_logger(__name__)


@dataclass
class RiskPrediction:
    """Risk classification prediction result."""
    risk_level: str
    confidence: float
    probabilities: Dict[str, float]


class RiskClassifier:
    """
    Classifies user input into risk categories (LOW/MEDIUM/HIGH).
    
    Uses a DistilRoBERTa model fine-tuned with LoRA for efficient
    risk classification.
    """
    
    def __init__(self, adapter_path: Optional[str] = None):
        """
        Initialize the risk classifier.
        
        Args:
            adapter_path: Path to LoRA adapter weights.
        """
        self.config = CONFIG.risk_classifier
        self.adapter_path = adapter_path or self.config.adapter_path
        self._model = None
        self._tokenizer = None
        self._initialized = False
        
        # Label mapping
        self.id2label = self.config.label_map
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # Set seeds
        set_all_seeds(CONFIG.inference.random_seed)
    
    def _load_model(self) -> bool:
        """
        Load the model and tokenizer.
        
        Returns:
            True if successful, False otherwise.
        """
        if self._initialized:
            return self._model is not None
        
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
            
            logger.info(f"Loading risk classifier: {self.config.base_model}")
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model
            )
            
            # Load model
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.config.base_model,
                num_labels=self.config.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
            )
            
            # Try to load LoRA adapter if available
            try:
                from peft import PeftModel
                import os
                
                if os.path.exists(self.adapter_path):
                    logger.info(f"Loading adapter from: {self.adapter_path}")
                    self._model = PeftModel.from_pretrained(
                        self._model,
                        self.adapter_path
                    )
                    logger.info("LoRA adapter loaded successfully")
                else:
                    logger.warning(
                        f"Adapter not found at {self.adapter_path}, "
                        "using base model (not fine-tuned)"
                    )
            except ImportError:
                logger.warning("PEFT not available, using base model")
            except Exception as e:
                logger.warning(f"Could not load adapter: {e}")
            
            self._model.eval()
            self._initialized = True
            
            logger.info("Risk classifier loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load risk classifier: {e}")
            self._initialized = True
            return False
    
    def _check_critical_keywords(self, text: str) -> Optional[RiskPrediction]:
        """
        MANDATORY safety check for critical crisis keywords.
        
        This runs BEFORE model prediction and CANNOT be overridden.
        If critical keywords are found, immediately returns HIGH risk.
        
        Args:
            text: User input text.
            
        Returns:
            RiskPrediction if crisis detected, None otherwise.
        """
        text_lower = text.lower()
        
        # Critical keywords that MUST trigger HIGH risk
        # These are immediate crisis indicators
        critical_keywords = [
            # Suicidal ideation
            "want to die", "wanna die", "kill myself", "end my life",
            "suicide", "suicidal", "end it all", "don't want to live",
            "no reason to live", "better off dead", "take my life",
            # Self-harm
            "hurt myself", "self-harm", "self harm", "overdose",
            "cut myself", "cutting myself", "hang myself",
            # Abuse/assault
            "assaulted", "raped", "molested", "abused", "beaten",
            "domestic violence", "sexually assaulted", "being abused",
        ]
        
        for keyword in critical_keywords:
            if keyword in text_lower:
                logger.warning(f"CRITICAL KEYWORD MATCHED: '{keyword}'")
                return RiskPrediction(
                    risk_level="HIGH",
                    confidence=1.0,
                    probabilities={"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 1.0}
                )
        
        return None
    
    def predict(self, text: str) -> RiskPrediction:
        """
        Predict risk level for input text.
        
        Args:
            text: User input text (in English).
            
        Returns:
            RiskPrediction with risk level and confidence.
        """
        # SAFETY FIRST: Mandatory crisis keyword check (cannot be overridden)
        crisis_result = self._check_critical_keywords(text)
        if crisis_result is not None:
            logger.warning(f"CRISIS KEYWORDS DETECTED: Forcing HIGH risk for safety")
            return crisis_result
        
        # Fallback for uninitialized model
        if not self._load_model():
            return self._keyword_based_prediction(text)
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Tokenize input
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)[0]
            
            # Get predicted class
            predicted_id = torch.argmax(probs).item()
            confidence = probs[predicted_id].item()
            predicted_label = self.id2label[predicted_id]
            
            # Build probability dict
            probabilities = {
                self.id2label[i]: probs[i].item()
                for i in range(len(self.id2label))
            }
            
            # Confidence threshold check for HIGH risk
            # Require high confidence (>0.7) for HIGH classification to prevent false positives
            if predicted_label == "HIGH" and confidence < 0.7:
                # Check if there are strong LOW signals
                if probabilities.get("LOW", 0) > 0.3:
                    predicted_label = "LOW"
                    confidence = probabilities["LOW"]
                    logger.info(f"Downgraded HIGH to LOW due to low confidence and strong LOW signal")
                else:
                    predicted_label = "MEDIUM"
                    confidence = probabilities.get("MEDIUM", confidence)
                    logger.info(f"Downgraded HIGH to MEDIUM due to low confidence ({confidence:.3f})")
            
            prediction = RiskPrediction(
                risk_level=predicted_label,
                confidence=confidence,
                probabilities=probabilities
            )
            
            logger.info(
                f"Risk prediction: {prediction.risk_level} "
                f"(confidence: {prediction.confidence:.3f}, probs: LOW={probabilities['LOW']:.2f}, MED={probabilities['MEDIUM']:.2f}, HIGH={probabilities['HIGH']:.2f})"
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._keyword_based_prediction(text)
    
    def _keyword_based_prediction(self, text: str) -> RiskPrediction:
        """
        Fallback keyword-based risk prediction.
        
        Used when model is not available.
        
        Args:
            text: User input text.
            
        Returns:
            RiskPrediction based on keyword matching.
        """
        text_lower = text.lower()
        
        # HIGH risk keywords
        high_risk_keywords = [
            "suicide", "kill myself", "end my life", "want to die",
            "don't want to live", "self-harm", "hurt myself",
            "end it all", "no reason to live", "better off dead",
            "overdose", "jump off", "hanging", "cut myself",
        ]
        
        # MEDIUM risk keywords
        medium_risk_keywords = [
            "hopeless", "worthless", "can't go on", "empty inside",
            "don't see the point", "give up", "no hope", "hate myself",
            "want to disappear", "burden", "failing at everything",
            "can't cope", "falling apart", "losing my mind",
        ]
        
        # Check for HIGH risk
        for keyword in high_risk_keywords:
            if keyword in text_lower:
                return RiskPrediction(
                    risk_level="HIGH",
                    confidence=0.9,
                    probabilities={"LOW": 0.05, "MEDIUM": 0.05, "HIGH": 0.9}
                )
        
        # Check for MEDIUM risk
        for keyword in medium_risk_keywords:
            if keyword in text_lower:
                return RiskPrediction(
                    risk_level="MEDIUM",
                    confidence=0.7,
                    probabilities={"LOW": 0.15, "MEDIUM": 0.7, "HIGH": 0.15}
                )
        
        # Default to LOW risk
        return RiskPrediction(
            risk_level="LOW",
            confidence=0.8,
            probabilities={"LOW": 0.8, "MEDIUM": 0.15, "HIGH": 0.05}
        )
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict risk levels for multiple texts.
        
        Args:
            texts: List of input texts.
            
        Returns:
            List of RiskPrediction objects.
        """
        return [self.predict(text) for text in texts]
    
    def is_high_risk(self, text: str) -> bool:
        """
        Quick check if input is high risk.
        
        Args:
            text: User input text.
            
        Returns:
            True if classified as HIGH risk.
        """
        prediction = self.predict(text)
        return prediction.risk_level == "HIGH"


# Singleton instance
_classifier_instance: Optional[RiskClassifier] = None


def get_risk_classifier() -> RiskClassifier:
    """Get the singleton risk classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = RiskClassifier()
    return _classifier_instance
