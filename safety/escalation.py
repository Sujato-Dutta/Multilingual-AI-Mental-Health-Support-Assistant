"""
Escalation Module

Provides predefined crisis responses and escalation handling
for HIGH-risk situations.
"""

from typing import Dict, Optional
from dataclasses import dataclass

from configs.prompts import ESCALATION_MESSAGES, get_escalation_message
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class EscalationResult:
    """Result of an escalation decision."""
    should_escalate: bool
    escalation_type: str
    response: str
    skip_generation: bool = True


class EscalationHandler:
    """
    Handles crisis escalation and predefined response selection.
    
    For HIGH-risk situations, this module provides deterministic,
    predefined responses that include crisis resources.
    """
    
    # Keywords that help determine escalation type
    ESCALATION_KEYWORDS = {
        "self_harm": [
            "cut", "cutting", "hurt myself", "hurting myself",
            "self-harm", "self harm", "harming myself",
            "burning myself", "hitting myself",
        ],
        "crisis_general": [
            "suicide", "suicidal", "kill myself", "end my life",
            "end it all", "don't want to live", "want to die",
            "no point living", "better off dead", "not be here",
            "disappear forever", "ending it",
        ],
        "abuse_disclosure": [
            "abusing me", "being abused", "hits me", "beats me",
            "molested", "raped", "sexually assaulted", "domestic violence",
            "hurts me physically", "afraid of my partner", "assaulted", "assault",
            "abused", "beaten", "attacked",
        ],
        "medical_emergency": [
            "overdose", "took too many", "swallowed", "poisoned",
            "can't breathe", "chest pain", "heart attack",
            "severe pain", "bleeding heavily", "unconscious",
        ],
    }
    
    def __init__(self):
        """Initialize the escalation handler."""
        logger.info("Escalation handler initialized")
    
    def determine_escalation_type(self, text: str) -> str:
        """
        Determine the type of escalation based on input text.
        
        Args:
            text: User's input text (in English).
            
        Returns:
            Escalation type string.
        """
        text_lower = text.lower()
        
        # Check for specific escalation types
        for escalation_type, keywords in self.ESCALATION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return escalation_type
        
        # Default to general crisis
        return "crisis_general"
    
    def get_escalation_response(
        self,
        risk_level: str,
        user_text: str
    ) -> EscalationResult:
        """
        Get appropriate escalation response for a risk level.
        
        Args:
            risk_level: Detected risk level (LOW/MEDIUM/HIGH).
            user_text: User's input text (in English).
            
        Returns:
            EscalationResult with escalation decision and response.
        """
        if risk_level != "HIGH":
            return EscalationResult(
                should_escalate=False,
                escalation_type="none",
                response="",
                skip_generation=False
            )
        
        # Determine specific escalation type
        escalation_type = self.determine_escalation_type(user_text)
        
        # Get predefined response
        response = get_escalation_message(escalation_type)
        
        logger.log_safety_event(
            event_type="escalation_triggered",
            risk_level=risk_level,
            action_taken="escalation_response",
            details=f"Type: {escalation_type}"
        )
        
        return EscalationResult(
            should_escalate=True,
            escalation_type=escalation_type,
            response=response,
            skip_generation=True
        )
    
    def should_skip_generation(self, risk_level: str) -> bool:
        """
        Determine if response generation should be skipped.
        
        For HIGH-risk cases, we use predefined responses instead
        of generated ones to ensure safety.
        
        Args:
            risk_level: Detected risk level.
            
        Returns:
            True if generation should be skipped.
        """
        return risk_level == "HIGH"
    
    def get_crisis_resources(self, country_code: str = "US") -> Dict[str, str]:
        """
        Get crisis resources for a specific country.
        
        Args:
            country_code: ISO country code.
            
        Returns:
            Dictionary of crisis resources.
        """
        # Currently focused on US resources
        # Can be extended for international support
        resources = {
            "US": {
                "suicide_prevention": "988 (Suicide & Crisis Lifeline)",
                "crisis_text": "Text HOME to 741741 (Crisis Text Line)",
                "domestic_violence": "1-800-799-7233 (National Domestic Violence Hotline)",
                "child_abuse": "1-800-422-4453 (Childhelp)",
                "sexual_assault": "1-800-656-4673 (RAINN)",
                "substance_abuse": "1-800-662-4357 (SAMHSA)",
                "emergency": "911",
            },
            "UK": {
                "samaritans": "116 123",
                "mind": "0300 123 3393",
                "crisis_text": "Text SHOUT to 85258",
                "emergency": "999",
            },
            "CA": {
                "crisis_line": "1-833-456-4566",
                "crisis_text": "Text 45645",
                "kids_help": "1-800-668-6868",
                "emergency": "911",
            },
        }
        
        return resources.get(country_code, resources["US"])
    
    def format_crisis_message(
        self,
        escalation_type: str,
        country_code: str = "US"
    ) -> str:
        """
        Format a crisis message with appropriate resources.
        
        Args:
            escalation_type: Type of escalation.
            country_code: ISO country code for resources.
            
        Returns:
            Formatted crisis message.
        """
        base_message = get_escalation_message(escalation_type)
        
        # For non-US countries, we could append localized resources
        # Currently returning base message which has US resources
        return base_message


# Singleton instance
_escalation_handler: Optional[EscalationHandler] = None


def get_escalation_handler() -> EscalationHandler:
    """Get the singleton escalation handler instance."""
    global _escalation_handler
    if _escalation_handler is None:
        _escalation_handler = EscalationHandler()
    return _escalation_handler
