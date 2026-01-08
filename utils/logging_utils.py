"""
Logging Utility Module

Provides structured logging with support for model versions,
adapter hashes, and configuration values.
"""

import logging
import os
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional


class StructuredLogger:
    """Logger with structured output support."""
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        log_dir: Optional[str] = None
    ):
        """
        Initialize the structured logger.
        
        Args:
            name: Logger name.
            level: Logging level.
            log_dir: Directory for log files (optional).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(
                log_dir,
                f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(console_format)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        self._log("info", message, kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        self._log("debug", message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        self._log("warning", message, kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        self._log("error", message, kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with optional structured data."""
        self._log("critical", message, kwargs)
    
    def _log(self, level: str, message: str, data: Dict[str, Any]):
        """Internal logging method."""
        log_method = getattr(self.logger, level)
        if data:
            data_str = json.dumps(data, default=str)
            log_method(f"{message} | {data_str}")
        else:
            log_method(message)
    
    def log_model_load(
        self,
        model_name: str,
        model_path: str,
        adapter_paths: Optional[list] = None,
        config_hash: Optional[str] = None
    ):
        """Log model loading event with version info."""
        self.info(
            f"Loading model: {model_name}",
            model_path=model_path,
            adapter_paths=adapter_paths,
            config_hash=config_hash
        )
    
    def log_inference(
        self,
        input_text: str,
        output_text: str,
        risk_level: str,
        language: str,
        latency_ms: float
    ):
        """Log inference event."""
        # Truncate long texts for logging
        input_preview = input_text[:100] + "..." if len(input_text) > 100 else input_text
        output_preview = output_text[:100] + "..." if len(output_text) > 100 else output_text
        
        self.info(
            "Inference completed",
            input_preview=input_preview,
            output_preview=output_preview,
            risk_level=risk_level,
            language=language,
            latency_ms=round(latency_ms, 2)
        )
    
    def log_safety_event(
        self,
        event_type: str,
        risk_level: str,
        action_taken: str,
        details: Optional[str] = None
    ):
        """Log safety-related events."""
        self.warning(
            f"Safety event: {event_type}",
            risk_level=risk_level,
            action_taken=action_taken,
            details=details
        )
    
    def log_translation(
        self,
        source_lang: str,
        target_lang: str,
        success: bool,
        error: Optional[str] = None
    ):
        """Log translation events."""
        if success:
            self.debug(
                "Translation completed",
                source_lang=source_lang,
                target_lang=target_lang
            )
        else:
            self.warning(
                "Translation failed",
                source_lang=source_lang,
                target_lang=target_lang,
                error=error
            )


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_dir: Optional[str] = None
) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name.
        level: Logging level (defaults to env var or INFO).
        log_dir: Directory for log files (optional).
        
    Returns:
        StructuredLogger instance.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    return StructuredLogger(name, level, log_dir)
