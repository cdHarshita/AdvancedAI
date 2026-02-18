"""
Security utilities for prompt injection protection and input validation.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of security validation."""
    is_valid: bool
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    issues: List[str]
    sanitized_input: Optional[str] = None


class PromptInjectionDetector:
    """
    Detect and prevent prompt injection attacks.
    Implements multiple layers of defense.
    """
    
    # Patterns that indicate potential prompt injection
    INJECTION_PATTERNS = [
        r'ignore\s+(previous|above|prior)\s+(instructions|directions|prompts)',
        r'disregard\s+(previous|above|prior)',
        r'forget\s+(everything|all)\s+(you|that)',
        r'new\s+(instructions|directions|task)',
        r'system\s*:\s*',
        r'<\s*script\s*>',
        r'<\s*\/\s*script\s*>',
        r'javascript\s*:',
        r'eval\s*\(',
        r'exec\s*\(',
        r'\bAI\b.*\bnow\b.*\byou\b',
        r'act\s+as\s+(?!assistant)',
        r'pretend\s+(?:you\s+are|to\s+be)',
        r'roleplay\s+as',
    ]
    
    # Suspicious keywords
    SUSPICIOUS_KEYWORDS = [
        'jailbreak', 'DAN', 'do anything now',
        'bypass', 'override', 'admin', 'root',
        'privilege', 'sudo', 'system prompt',
        'developer mode', 'god mode'
    ]
    
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
    
    def detect(self, user_input: str) -> ValidationResult:
        """
        Detect potential prompt injection attempts.
        
        Args:
            user_input: User-provided input
        
        Returns:
            ValidationResult with detection details
        """
        issues = []
        risk_level = "LOW"
        
        # Check for injection patterns
        for pattern in self.patterns:
            if pattern.search(user_input):
                issues.append(f"Detected potential injection pattern: {pattern.pattern}")
                risk_level = "HIGH"
        
        # Check for suspicious keywords
        lower_input = user_input.lower()
        for keyword in self.SUSPICIOUS_KEYWORDS:
            if keyword.lower() in lower_input:
                issues.append(f"Detected suspicious keyword: {keyword}")
                if risk_level == "LOW":
                    risk_level = "MEDIUM"
        
        # Check for excessive special characters
        special_char_ratio = len([c for c in user_input if not c.isalnum() and not c.isspace()]) / max(len(user_input), 1)
        if special_char_ratio > 0.3:
            issues.append("Excessive special characters detected")
            if risk_level == "LOW":
                risk_level = "MEDIUM"
        
        # Check for encoding attempts
        if re.search(r'\\x[0-9a-f]{2}|\\u[0-9a-f]{4}|&#\d+;', user_input, re.IGNORECASE):
            issues.append("Detected encoded characters")
            risk_level = "HIGH"
        
        is_valid = risk_level in ["LOW", "MEDIUM"]
        
        if not is_valid:
            logger.warning(f"Potential prompt injection detected: {issues}")
        
        return ValidationResult(
            is_valid=is_valid,
            risk_level=risk_level,
            issues=issues,
            sanitized_input=self.sanitize(user_input) if is_valid else None
        )
    
    def sanitize(self, user_input: str, max_length: int = 4000) -> str:
        """
        Sanitize user input.
        
        Args:
            user_input: User-provided input
            max_length: Maximum allowed length
        
        Returns:
            Sanitized input
        """
        # Truncate to max length
        sanitized = user_input[:max_length]
        
        # Remove potentially dangerous HTML/script tags
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'<iframe[^>]*>.*?</iframe>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized


class InputValidator:
    """Validate and sanitize various input types."""
    
    @staticmethod
    def validate_json_input(data: Dict[str, Any], max_depth: int = 5) -> ValidationResult:
        """
        Validate JSON input for safety.
        
        Args:
            data: JSON data to validate
            max_depth: Maximum allowed nesting depth
        
        Returns:
            ValidationResult
        """
        issues = []
        
        def check_depth(obj: Any, current_depth: int = 0) -> int:
            if current_depth > max_depth:
                return current_depth
            
            if isinstance(obj, dict):
                return max(check_depth(v, current_depth + 1) for v in obj.values()) if obj else current_depth
            elif isinstance(obj, list):
                return max(check_depth(v, current_depth + 1) for v in obj) if obj else current_depth
            return current_depth
        
        depth = check_depth(data)
        if depth > max_depth:
            issues.append(f"JSON depth {depth} exceeds maximum {max_depth}")
            return ValidationResult(is_valid=False, risk_level="HIGH", issues=issues)
        
        return ValidationResult(is_valid=True, risk_level="LOW", issues=[])
    
    @staticmethod
    def validate_file_path(path: str) -> ValidationResult:
        """
        Validate file paths to prevent path traversal.
        
        Args:
            path: File path to validate
        
        Returns:
            ValidationResult
        """
        issues = []
        
        # Check for path traversal attempts
        if '..' in path or path.startswith('/'):
            issues.append("Path traversal attempt detected")
            return ValidationResult(is_valid=False, risk_level="CRITICAL", issues=issues)
        
        # Check for suspicious characters
        if re.search(r'[<>:"|?*]', path):
            issues.append("Invalid characters in file path")
            return ValidationResult(is_valid=False, risk_level="HIGH", issues=issues)
        
        return ValidationResult(is_valid=True, risk_level="LOW", issues=[])


def validate_input(
    user_input: str,
    max_length: int = 4000,
    check_injection: bool = True
) -> ValidationResult:
    """
    Comprehensive input validation.
    
    Args:
        user_input: User-provided input
        max_length: Maximum allowed length
        check_injection: Whether to check for prompt injection
    
    Returns:
        ValidationResult
    """
    issues = []
    risk_level = "LOW"
    
    # Length check
    if len(user_input) > max_length:
        issues.append(f"Input length {len(user_input)} exceeds maximum {max_length}")
        risk_level = "MEDIUM"
        user_input = user_input[:max_length]
    
    # Prompt injection check
    if check_injection:
        detector = PromptInjectionDetector()
        injection_result = detector.detect(user_input)
        
        if not injection_result.is_valid:
            return injection_result
        
        if injection_result.risk_level == "MEDIUM":
            risk_level = "MEDIUM"
            issues.extend(injection_result.issues)
    
    return ValidationResult(
        is_valid=True,
        risk_level=risk_level,
        issues=issues,
        sanitized_input=user_input
    )
