"""
Unified Inference Pipeline

Orchestrates the complete flow: translation, risk detection,
response generation, safety checks, and translation back.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time

from configs.model_config import CONFIG
from utils.logging_utils import get_logger
from utils.seed import set_all_seeds

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    """A single conversation turn."""
    role: str
    content: str
    original_content: Optional[str] = None
    language: str = "en"


@dataclass
class PipelineResult:
    """Result of pipeline inference."""
    response: str
    original_language: str
    risk_level: str
    was_escalated: bool
    was_safety_overridden: bool
    latency_ms: float
    translation_success: bool = True


class InferencePipeline:
    """
    Complete inference pipeline for the mental health assistant.
    
    Manages: language detection, translation, risk classification,
    response generation, safety guardrails, and conversation memory.
    """
    
    def __init__(self):
        """Initialize the inference pipeline."""
        self._translator = None
        self._risk_classifier = None
        self._response_generator = None
        self._safety_guardrails = None
        self._escalation_handler = None
        
        self.conversation_history: List[ConversationTurn] = []
        self.max_turns = CONFIG.inference.max_conversation_turns
        
        set_all_seeds(CONFIG.inference.random_seed)
        logger.info("Inference pipeline initialized")
    
    @property
    def translator(self):
        """Lazy load translator."""
        if self._translator is None:
            from translation.translator import Translator
            self._translator = Translator()
        return self._translator
    
    @property
    def risk_classifier(self):
        """Lazy load risk classifier."""
        if self._risk_classifier is None:
            from models.risk_classifier import get_risk_classifier
            self._risk_classifier = get_risk_classifier()
        return self._risk_classifier
    
    @property
    def response_generator(self):
        """Lazy load response generator."""
        if self._response_generator is None:
            from models.response_generator import get_response_generator
            self._response_generator = get_response_generator()
        return self._response_generator
    
    @property
    def safety_guardrails(self):
        """Lazy load safety guardrails."""
        if self._safety_guardrails is None:
            from safety.guardrails import SafetyGuardrails
            self._safety_guardrails = SafetyGuardrails()
        return self._safety_guardrails
    
    @property
    def escalation_handler(self):
        """Lazy load escalation handler."""
        if self._escalation_handler is None:
            from safety.escalation import get_escalation_handler
            self._escalation_handler = get_escalation_handler()
        return self._escalation_handler
    
    def process_text(self, user_input: str) -> PipelineResult:
        """
        Process text input through the full pipeline.
        
        Args:
            user_input: User's text input.
            
        Returns:
            PipelineResult with response and metadata.
        """
        start_time = time.time()
        
        if not user_input or not user_input.strip():
            return PipelineResult(
                response="I'm here to listen. What would you like to talk about?",
                original_language="en",
                risk_level="LOW",
                was_escalated=False,
                was_safety_overridden=False,
                latency_ms=0
            )
        
        # Step 1: Language detection and translation to English
        english_text, detected_lang, trans_success = self.translator.process_input(user_input)
        
        # Step 2: Risk classification (on latest message only)
        risk_pred = self.risk_classifier.predict(english_text)
        risk_level = risk_pred.risk_level
        
        # Step 3: Check for escalation (HIGH risk)
        escalation = self.escalation_handler.get_escalation_response(risk_level, english_text)
        
        if escalation.should_escalate:
            # Use predefined escalation response
            response_english = escalation.response
            was_escalated = True
            was_overridden = False
        else:
            # Step 4: Generate response
            history = self._get_history_for_generation()
            gen_result = self.response_generator.generate(
                english_text,
                history,
                deterministic=(risk_level == "MEDIUM")
            )
            response_english = gen_result.response
            
            # Step 5: Safety guardrail check
            response_english, was_overridden = self.safety_guardrails.validate_and_fix(
                response_english, risk_level
            )
            was_escalated = False
        
        # Step 6: Translate response back to original language
        # NOTE: Numbers, URLs, and phone numbers are protected during translation
        if detected_lang != "en":
            final_response, _ = self.translator.translate_from_english(
                response_english, detected_lang
            )
        else:
            final_response = response_english
        
        # Update conversation history
        self._add_to_history(user_input, detected_lang, final_response, response_english)
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.log_inference(
            input_text=user_input,
            output_text=final_response,
            risk_level=risk_level,
            language=detected_lang,
            latency_ms=latency_ms
        )
        
        return PipelineResult(
            response=final_response,
            original_language=detected_lang,
            risk_level=risk_level,
            was_escalated=was_escalated,
            was_safety_overridden=was_overridden,
            latency_ms=latency_ms,
            translation_success=trans_success
        )
    
    def _get_history_for_generation(self) -> List[Dict[str, str]]:
        """Get conversation history formatted for the model."""
        history = []
        for turn in self.conversation_history[-self.max_turns * 2:]:
            # Use English content for generation
            content = turn.original_content if turn.original_content else turn.content
            history.append({"role": turn.role, "content": content})
        return history
    
    def _add_to_history(
        self,
        user_input: str,
        language: str,
        response: str,
        english_response: str
    ) -> None:
        """Add a conversation turn to history."""
        # User turn
        english_input, _, _ = self.translator.process_input(user_input)
        self.conversation_history.append(ConversationTurn(
            role="user",
            content=user_input,
            original_content=english_input,
            language=language
        ))
        
        # Assistant turn
        self.conversation_history.append(ConversationTurn(
            role="assistant",
            content=response,
            original_content=english_response,
            language=language
        ))
        
        # Trim history if needed
        max_entries = self.max_turns * 2
        if len(self.conversation_history) > max_entries:
            self.conversation_history = self.conversation_history[-max_entries:]
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> Dict:
        """Get a summary of the current conversation."""
        return {
            "turn_count": len(self.conversation_history) // 2,
            "max_turns": self.max_turns,
            "languages_used": list(set(t.language for t in self.conversation_history))
        }


_pipeline_instance: Optional[InferencePipeline] = None


def get_pipeline() -> InferencePipeline:
    """Get the singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = InferencePipeline()
    return _pipeline_instance
