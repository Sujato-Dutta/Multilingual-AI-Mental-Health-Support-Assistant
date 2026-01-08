"""
Response Generator Module

Provides response generation using Qwen2.5-3B-Instruct with QLoRA
and adapter fusion for instruction-tuned and safety-aware responses.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import time

from configs.model_config import CONFIG
from configs.prompts import get_system_prompt
from utils.logging_utils import get_logger
from utils.seed import set_all_seeds

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """Result of response generation."""
    response: str
    input_tokens: int
    output_tokens: int
    generation_time_ms: float


class ResponseGenerator:
    """Generates supportive responses using a fine-tuned Qwen model."""
    
    FALLBACK_RESPONSES = [
        "I hear you, and I'm here to listen. Would you like to tell me more?",
        "Thank you for sharing. It sounds difficult. How can I support you?",
        "Your feelings are valid. I'm here without judgment. What's on your mind?",
    ]
    
    def __init__(
        self,
        instruction_adapter_path: Optional[str] = None,
        safety_adapter_path: Optional[str] = None
    ):
        """Initialize the response generator."""
        self.config = CONFIG.response_generator
        self.qlora_config = CONFIG.qlora
        self.instruction_adapter_path = instruction_adapter_path or self.config.instruction_adapter_path
        self.safety_adapter_path = safety_adapter_path or self.config.safety_adapter_path
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._fallback_idx = 0
        set_all_seeds(CONFIG.inference.random_seed)
    
    def _load_model(self) -> bool:
        """Load the model with QLoRA quantization and adapters."""
        if self._initialized:
            return self._model is not None
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading: {self.config.base_model}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model, trust_remote_code=True
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float32, "low_cpu_mem_usage": True}
            
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=self.qlora_config.load_in_4bit,
                    bnb_4bit_quant_type=self.qlora_config.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=getattr(torch, self.qlora_config.bnb_4bit_compute_dtype),
                    bnb_4bit_use_double_quant=self.qlora_config.bnb_4bit_use_double_quant,
                )
                model_kwargs["quantization_config"] = bnb_config
            except ImportError:
                pass
            
            self._model = AutoModelForCausalLM.from_pretrained(self.config.base_model, **model_kwargs)
            self._load_adapters()
            self._model.eval()
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to load: {e}")
            self._initialized = True
            return False
    
    def _load_adapters(self) -> None:
        """Load and fuse LoRA adapters if available."""
        try:
            from peft import PeftModel
            import os
            
            adapters = []
            if os.path.exists(self.instruction_adapter_path):
                self._model = PeftModel.from_pretrained(self._model, self.instruction_adapter_path, adapter_name="instruction")
                adapters.append("instruction")
            if os.path.exists(self.safety_adapter_path):
                if adapters:
                    self._model.load_adapter(self.safety_adapter_path, adapter_name="safety")
                else:
                    self._model = PeftModel.from_pretrained(self._model, self.safety_adapter_path, adapter_name="safety")
                adapters.append("safety")
            
            if len(adapters) == 2:
                self._model.add_weighted_adapter(adapters, [0.4, 0.6], "merged", "linear")
                self._model.set_adapter("merged")
            elif adapters:
                self._model.set_adapter(adapters[0])
        except (ImportError, Exception):
            pass
    
    def _get_fallback(self) -> str:
        resp = self.FALLBACK_RESPONSES[self._fallback_idx]
        self._fallback_idx = (self._fallback_idx + 1) % len(self.FALLBACK_RESPONSES)
        return resp
    
    def _build_messages(self, user_msg: str, history: Optional[List[Dict]] = None) -> List[Dict]:
        msgs = [{"role": "system", "content": get_system_prompt()}]
        if history:
            msgs.extend(history[-CONFIG.inference.max_conversation_turns * 2:])
        msgs.append({"role": "user", "content": user_msg})
        return msgs
    
    def _format_prompt(self, messages: List[Dict]) -> str:
        if hasattr(self._tokenizer, 'apply_chat_template'):
            try:
                return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        parts = []
        for m in messages:
            if m["role"] == "system":
                parts.append(f"System: {m['content']}\n")
            elif m["role"] == "user":
                parts.append(f"User: {m['content']}\n")
            else:
                parts.append(f"Assistant: {m['content']}\n")
        parts.append("Assistant: ")
        return "".join(parts)
    
    def _extract_response(self, full: str, prompt: str) -> str:
        """Extract only the assistant's response from the full generated output."""
        # First, remove the prompt if it's at the beginning
        if full.startswith(prompt):
            resp = full[len(prompt):].strip()
        else:
            resp = full.strip()
        
        # Look for the last "assistant" marker and extract after it
        # This handles Qwen's chat template format
        lower_resp = resp.lower()
        
        # Try multiple assistant markers
        markers = ["assistant\n", "assistant:", "<|assistant|>", "assistant"]
        best_pos = -1
        for marker in markers:
            pos = lower_resp.rfind(marker)
            if pos > best_pos:
                best_pos = pos
                # Extract text after the marker
                marker_len = len(marker)
                resp = resp[pos + marker_len:].strip()
        
        # If "system" appears at the start, the extraction failed - use fallback
        if resp.lower().startswith("system") or resp.lower().startswith("you are"):
            # Fallback: just use the last sentence or a safe default
            logger.warning("Response extraction failed, using fallback")
            return self._get_fallback()
        
        # Clean up stop sequences
        stop_sequences = ["User:", "Human:", "user:", "human:", "<|", "\n\n\n", "System:", "system:"]
        for stop in stop_sequences:
            if stop in resp:
                resp = resp[:resp.index(stop)].strip()
        
        # Final cleanup - remove any leading/trailing special chars
        resp = resp.strip('"\' \n\t')
        
        # If response is empty or too short, use fallback
        if len(resp) < 5:
            return self._get_fallback()
        
        return resp
    
    def generate(self, user_msg: str, history: Optional[List[Dict]] = None, deterministic: bool = False) -> GenerationResult:
        """Generate a response to the user message."""
        start = time.time()
        if not self._load_model():
            return GenerationResult(self._get_fallback(), 0, 0, 0)
        
        try:
            import torch
            msgs = self._build_messages(user_msg, history)
            prompt = self._format_prompt(msgs)
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_seq_length)
            in_toks = inputs.input_ids.shape[1]
            
            gen_kwargs = {"max_new_tokens": self.config.max_new_tokens, "pad_token_id": self._tokenizer.pad_token_id,
                          "eos_token_id": self._tokenizer.eos_token_id, "repetition_penalty": self.config.repetition_penalty}
            if deterministic:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs.update({"do_sample": True, "temperature": self.config.temperature, "top_p": self.config.top_p})
            
            with torch.no_grad():
                out = self._model.generate(**inputs, **gen_kwargs)
            
            full = self._tokenizer.decode(out[0], skip_special_tokens=True)
            resp = self._extract_response(full, prompt)
            out_toks = out.shape[1] - in_toks
            
            return GenerationResult(resp, in_toks, out_toks, (time.time() - start) * 1000)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return GenerationResult(self._get_fallback(), 0, 0, (time.time() - start) * 1000)


_generator: Optional[ResponseGenerator] = None

def get_response_generator() -> ResponseGenerator:
    global _generator
    if _generator is None:
        _generator = ResponseGenerator()
    return _generator
