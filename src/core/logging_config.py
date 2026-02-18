"""
Comprehensive logging setup with structured logging and security filtering.
"""

import logging
import sys
import json
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
from pythonjsonlogger import jsonlogger
import re


class SecurityFilter(logging.Filter):
    """Filter to prevent logging sensitive information."""
    
    SENSITIVE_PATTERNS = [
        r'api[_-]?key["\s:=]+[\w-]+',
        r'secret["\s:=]+[\w-]+',
        r'password["\s:=]+[\w-]+',
        r'token["\s:=]+[\w-]+',
        r'authorization["\s:=]+[\w-]+',
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive information from log messages."""
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            for pattern in self.SENSITIVE_PATTERNS:
                msg = re.sub(pattern, '[REDACTED]', msg, flags=re.IGNORECASE)
            record.msg = msg
        return True


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional context."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/ai_systems.log",
    log_format: str = "json",
    enable_console: bool = True
) -> logging.Logger:
    """
    Setup structured logging with security filtering.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_format: Format type ('json' or 'text')
        enable_console: Whether to enable console output
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    logger = logging.getLogger("ai_systems")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Add security filter
    security_filter = SecurityFilter()
    
    # File handler with rotation
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    file_handler.addFilter(security_filter)
    
    if log_format == "json":
        file_formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
    else:
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.addFilter(security_filter)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"ai_systems.{name}")
