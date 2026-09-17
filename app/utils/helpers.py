import os
import re
import json
import logging
from typing import Dict, Any, Optional

def setup_app_logging(name: str = "DocVisionAI", level: str = "INFO") -> logging.Logger:
    """Sets up a standardized logger for application modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def extract_json_block(text: str) -> Dict[str, Any]:
    """
    Finds and parses a JSON block within text.
    Handles typical markdown wraps (```json ... ```).
    """
    text_clean = text.strip()
    
    # Check for markdown code block wraps
    if "```json" in text_clean:
        text_clean = text_clean.split("```json")[1].split("```")[0].strip()
    elif "```" in text_clean:
        text_clean = text_clean.split("```")[1].split("```")[0].strip()
        
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        # Search for first '{' and last '}'
        match = re.search(r"\{.*\}", text_clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"error": "Failed to parse JSON content from text", "raw_content": text}

def normalize_text_spacing(text: str) -> str:
    """Standardizes string spacings and case for comparison tasks."""
    return " ".join(text.strip().split())
