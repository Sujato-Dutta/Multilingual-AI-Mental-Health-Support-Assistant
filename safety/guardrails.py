"""
Safety Guardrails Module

Provides post-generation safety checks and output validation
to ensure responses meet safety standards.
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

from configs.prompts import get_safe_refusal, SAFE_REFUSALS
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""
    is_safe: bool
    violation_type: Optional[str] = None
    confidence: float = 1.0
    override_response: Optional[str] = None


class SafetyGuardrails:
    """
    Post-generation safety layer for validating and filtering responses.
    
    Performs deterministic safety checks on generated outputs and can
    override unsafe content with predefined safe responses.
    """
    
    # Patterns that indicate potentially unsafe content
    UNSAFE_PATTERNS = {
        "medication_recommendation": [
            r"you should take\s+\w+",
            r"try taking\s+\w+\s+mg",
            r"i recommend\s+\w+\s+medication",
            r"take\s+\d+\s*mg\s+of",
            r"prescription for\s+\w+",
            r"dosage of\s+\d+",
        ],
        "diagnosis_claim": [
            r"you have\s+(depression|anxiety|bipolar|ptsd|schizophrenia|ocd)",
            r"you are\s+(depressed|bipolar|schizophrenic)",
            r"you're suffering from\s+\w+\s+disorder",
            r"i diagnose you",
            r"your diagnosis is",
            r"based on what you've told me,?\s+you have",
        ],
        "therapy_claim": [
            r"as your therapist",
            r"in our therapy session",
            r"therapeutic intervention",
            r"i am providing therapy",
            r"this is a therapy session",
        ],
        "harmful_advice": [
            r"you should hurt",
            r"go ahead and\s+(cut|harm|hurt)",
            r"it's okay to\s+(cut|harm|hurt)\s+yourself",
            r"methods to\s+(die|kill|end)",
            r"how to\s+(commit suicide|kill yourself|end your life)",
            r"pills to take to",
        ],
        "minimizing_crisis": [
            r"don't be dramatic",
            r"you're overreacting",
            r"it's not that bad",
            r"just get over it",
            r"stop being so negative",
            r"you're fine",
            r"there's nothing wrong with you",
        ],
    }
    
    # Keywords that may indicate incomplete safety responses
    REQUIRED_CRISIS_ELEMENTS = [
        "988",
        "crisis",
        "help",
        "support",
    ]
    
    def __init__(self):
        """Initialize the safety guardrails."""
        self._compile_patterns()
        logger.info("Safety guardrails initialized")
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        self._compiled_patterns = {}
        for category, patterns in self.UNSAFE_PATTERNS.items():
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def check_output_safety(
        self,
        response: str,
        risk_level: str = "LOW"
    ) -> SafetyCheckResult:
        """
        Check if a generated response is safe.
        
        Args:
            response: The generated response text.
            risk_level: The detected risk level of the user's input.
            
        Returns:
            SafetyCheckResult with safety assessment.
        """
        if not response or not response.strip():
            return SafetyCheckResult(is_safe=True)
        
        response_lower = response.lower()
        
        # Check for unsafe patterns
        for category, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(response):
                    logger.log_safety_event(
                        event_type="unsafe_pattern_detected",
                        risk_level=risk_level,
                        action_taken="flagged",
                        details=f"Category: {category}"
                    )
                    return SafetyCheckResult(
                        is_safe=False,
                        violation_type=category,
                        override_response=self._get_override_for_category(category)
                    )
        
        # For HIGH risk, verify crisis resources are mentioned
        if risk_level == "HIGH":
            has_crisis_elements = any(
                element in response_lower
                for element in self.REQUIRED_CRISIS_ELEMENTS
            )
            if not has_crisis_elements:
                logger.log_safety_event(
                    event_type="missing_crisis_resources",
                    risk_level=risk_level,
                    action_taken="flagged",
                    details="HIGH risk response missing crisis resources"
                )
                return SafetyCheckResult(
                    is_safe=False,
                    violation_type="missing_crisis_resources",
                    confidence=0.8
                )
        
        return SafetyCheckResult(is_safe=True)
    
    def _get_override_for_category(self, category: str) -> str:
        """Get override response for a violation category."""
        override_mapping = {
            "medication_recommendation": "medical_advice",
            "diagnosis_claim": "diagnosis_request",
            "therapy_claim": "therapy_request",
            "harmful_advice": "harmful_request",
            "minimizing_crisis": "harmful_request",
        }
        refusal_type = override_mapping.get(category, "harmful_request")
        return get_safe_refusal(refusal_type)
    
    def apply_safety_override(
        self,
        response: str,
        check_result: SafetyCheckResult
    ) -> str:
        """
        Apply safety override if needed.
        
        Args:
            response: Original response.
            check_result: Result from safety check.
            
        Returns:
            Safe response (either original or override).
        """
        if check_result.is_safe:
            return response
        
        if check_result.override_response:
            logger.log_safety_event(
                event_type="response_overridden",
                risk_level="UNKNOWN",
                action_taken="replaced",
                details=f"Violation: {check_result.violation_type}"
            )
            return check_result.override_response
        
        # Default safe fallback
        return SAFE_REFUSALS["harmful_request"]
    
    def validate_and_fix(
        self,
        response: str,
        risk_level: str = "LOW"
    ) -> Tuple[str, bool]:
        """
        Validate response and apply fixes if needed.
        
        Args:
            response: Generated response.
            risk_level: Detected risk level.
            
        Returns:
            Tuple of (final_response, was_modified).
        """
        check_result = self.check_output_safety(response, risk_level)
        
        if check_result.is_safe:
            return response, False
        
        fixed_response = self.apply_safety_override(response, check_result)
        return fixed_response, True
    
    def filter_sensitive_content(self, text: str) -> str:
        """
        Filter potentially sensitive content from text.
        
        Args:
            text: Text to filter.
            
        Returns:
            Filtered text.
        """
        # Remove potential PII patterns
        # Phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        # Email addresses
        text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', text)
        # SSN patterns
        text = re.sub(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', '[SSN]', text)
        
        return text
