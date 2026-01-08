"""
Translation Module

Provides language detection and bidirectional translation capabilities
using langdetect and argos-translate libraries.
"""

import logging
import re
from typing import Optional, Tuple, List, Dict

from configs.model_config import CONFIG
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class Translator:
    """
    Handles language detection and translation for the mental health assistant.
    
    Uses langdetect for language identification and argos-translate for
    bidirectional translation between English and supported languages.
    """
    
    def __init__(self):
        """Initialize the translator with required packages."""
        # Fix for Streamlit Cloud permissions: Set local package directory
        import os
        from pathlib import Path
        
        # Create a local cache directory for argos packages
        cache_dir = Path("data/argos_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variable BEFORE importing argostranslate
        os.environ["ARGOS_PACKAGES_DIR"] = str(cache_dir.absolute())
        
        self._langdetect_available = False
        self._argos_available = False
        self._installed_languages = set()
        self._translation_cache = {}
        
        self._initialize_langdetect()
        self._initialize_argos()
    
    def _initialize_langdetect(self) -> None:
        """Initialize langdetect library."""
        try:
            from langdetect import detect, DetectorFactory
            # Set seed for deterministic language detection
            DetectorFactory.seed = CONFIG.inference.random_seed
            self._langdetect_available = True
            logger.info("langdetect initialized successfully")
        except ImportError:
            logger.warning("langdetect not available - language detection disabled")
    
    def _initialize_argos(self) -> None:
        """Initialize argos-translate and download required language packages."""
        try:
            import argostranslate.package
            import argostranslate.translate
            
            # Update package index
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            
            # Get installed languages
            installed_packages = argostranslate.package.get_installed_packages()
            for pkg in installed_packages:
                self._installed_languages.add((pkg.from_code, pkg.to_code))
            
            self._argos_available = True
            logger.info(
                "argos-translate initialized",
                installed_language_pairs=len(self._installed_languages)
            )
        except ImportError:
            logger.warning("argos-translate not available - translation disabled")
        except Exception as e:
            logger.error(f"Error initializing argos-translate: {e}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            ISO 639-1 language code (e.g., 'en', 'es', 'fr').
            Returns 'en' as fallback if detection fails.
        """
        if not self._langdetect_available:
            return CONFIG.translation.fallback_language
        
        if not text or not text.strip():
            return CONFIG.translation.fallback_language
        
        try:
            from langdetect import detect
            detected = detect(text)
            logger.debug(f"Detected language: {detected}")
            return detected
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return CONFIG.translation.fallback_language
    
    def _ensure_package_installed(self, from_code: str, to_code: str) -> bool:
        """
        Ensure the translation package for a language pair is installed.
        
        Args:
            from_code: Source language code.
            to_code: Target language code.
            
        Returns:
            True if package is available, False otherwise.
        """
        if (from_code, to_code) in self._installed_languages:
            return True
        
        if not self._argos_available:
            return False
        
        try:
            import argostranslate.package
            
            available_packages = argostranslate.package.get_available_packages()
            package_to_install = next(
                (pkg for pkg in available_packages 
                 if pkg.from_code == from_code and pkg.to_code == to_code),
                None
            )
            
            if package_to_install:
                logger.info(f"Installing translation package: {from_code} -> {to_code}")
                argostranslate.package.install_from_path(package_to_install.download())
                self._installed_languages.add((from_code, to_code))
                return True
            else:
                logger.warning(f"No package available for: {from_code} -> {to_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to install translation package: {e}")
            return False
    
    def _translate_with_protection(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text while preserving sensitive content by splitting and translating segments.
        
        Args:
            text: Text to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            
        Returns:
            Translated text with sensitive content preserved.
        """
        import argostranslate.translate
        
        # Combined pattern for sensitive content
        # Order matters: URLs first, then long phone numbers, then short codes
        pattern = r'(https?://[^\s\)]+|\b\d[\d\-\s]{6,}\d\b|\b\d{3,6}\b)'
        
        # Split text by pattern, preserving separators (sensitive content)
        segments = re.split(pattern, text)
        
        translated_segments = []
        for segment in segments:
            # Check if segment matches the sensitive pattern
            if re.match(pattern, segment):
                # Keep sensitive content exact
                translated_segments.append(segment)
            elif segment.strip():
                # Translate text content
                try:
                    trans = argostranslate.translate.translate(segment, source_lang, target_lang)
                    translated_segments.append(trans)
                except Exception:
                    translated_segments.append(segment)
            else:
                # Keep whitespace/empty segments
                translated_segments.append(segment)
                
        return "".join(translated_segments)
    
    def translate_to_english(self, text: str, source_lang: str) -> Tuple[str, bool]:
        """
        Translate text from source language to English.
        
        Args:
            text: Text to translate.
            source_lang: Source language code.
            
        Returns:
            Tuple of (translated_text, success_flag).
            Returns original text with False if translation fails.
        """
        if source_lang == "en":
            return text, True
        
        if not self._argos_available:
            logger.warning("Translation not available, returning original text")
            return text, False
            
        # Ensure package is installed
        if not self._ensure_package_installed(source_lang, "en"):
            logger.log_translation(source_lang, "en", False, "Package not available")
            return text, False
        
        try:
            # Check cache
            cache_key = (text, source_lang, "en")
            if cache_key in self._translation_cache:
                return self._translation_cache[cache_key], True
            
            # Use segmentation protection strategy
            translated = self._translate_with_protection(text, source_lang, "en")
            
            # Cache and return
            self._translation_cache[cache_key] = translated
            logger.log_translation(source_lang, "en", True)
            return translated, True
            
        except Exception as e:
            logger.log_translation(source_lang, "en", False, str(e))
            return text, False

    def translate_from_english(self, text: str, target_lang: str) -> Tuple[str, bool]:
        """
        Translate text from English to target language.
        
        Args:
            text: Text to translate (in English).
            target_lang: Target language code.
            
        Returns:
            Tuple of (translated_text, success_flag).
            Returns original text with False if translation fails.
        """
        if target_lang == "en":
            return text, True
        
        if not self._argos_available:
            logger.warning("Translation not available, returning original text")
            return text, False
            
        # Ensure package is installed
        if not self._ensure_package_installed("en", target_lang):
            logger.log_translation("en", target_lang, False, "Package not available")
            return text, False
        
        try:
            # Check cache
            cache_key = (text, "en", target_lang)
            if cache_key in self._translation_cache:
                return self._translation_cache[cache_key], True
            
            # Use segmentation protection strategy
            translated = self._translate_with_protection(text, "en", target_lang)
            
            # Cache and return
            self._translation_cache[cache_key] = translated
            logger.log_translation("en", target_lang, True)
            return translated, True
            
        except Exception as e:
            logger.log_translation("en", target_lang, False, str(e))
            return text, False
    
    def process_input(self, text: str) -> Tuple[str, str, bool]:
        """
        Process user input: detect language and translate to English.
        
        Args:
            text: User's input text.
            
        Returns:
            Tuple of (english_text, original_language, translation_success).
        """
        if not text or not text.strip():
            return "", "en", True
        
        # Detect language
        detected_lang = self.detect_language(text)
        
        # Translate to English if needed
        english_text, success = self.translate_to_english(text, detected_lang)
        
        return english_text, detected_lang, success
    
    def process_output(self, text: str, target_lang: str) -> Tuple[str, bool]:
        """
        Process assistant output: translate from English to target language.
        
        Args:
            text: Assistant's response in English.
            target_lang: Target language for translation.
            
        Returns:
            Tuple of (translated_text, translation_success).
        """
        if not text or not text.strip():
            return "", True
        
        return self.translate_from_english(text, target_lang)
    
    def is_supported_language(self, lang_code: str) -> bool:
        """
        Check if a language is supported for translation.
        
        Args:
            lang_code: ISO 639-1 language code.
            
        Returns:
            True if language is supported.
        """
        return lang_code in CONFIG.translation.supported_languages
    
    def clear_cache(self) -> None:
        """Clear the translation cache."""
        self._translation_cache.clear()
        logger.debug("Translation cache cleared")
