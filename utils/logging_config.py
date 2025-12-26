"""
Logging Configuration - Logging setup for the RAG pipeline.
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Where to store log files
LOG_DIR = Path("logs")


def setup_logging(level: str = "INFO"):
    """
    Set up logging for the application.
    
    Args:
        level: Log level - "DEBUG", "INFO", "WARNING", "ERROR"
        
    Returns:
        The root logger
    """
    # Create logs folder if it doesn't exist
    LOG_DIR.mkdir(exist_ok=True)
    
    # Convert level string to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Format for log messages
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler - prints to terminal
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler - saves to file
    log_file = LOG_DIR / f"rag_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Always save debug to file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to file: {log_file}")
    
    return logger


def get_logger(name: str):
    """Get a logger with a specific name."""
    return logging.getLogger(name)


class PipelineLogger:
    """
    Simple logger for tracking RAG pipeline metrics.
    
    Usage:
        logger = PipelineLogger("my_app")
        logger.log_retrieval("user query", docs_count=5, time=0.5)
        logger.log_generation(test_cases=3, tokens=500, time=2.0)
    """
    
    def __init__(self, name: str = "rag_pipeline"):
        self.logger = logging.getLogger(name)
    
    def log_retrieval(self, query: str, docs_retrieved: int, docs_after_filter: int, 
                      avg_relevance: float, duration_seconds: float):
        """Log retrieval step metrics."""
        self.logger.info(
            f"[RETRIEVAL] Query: '{query[:50]}...' | "
            f"Docs: {docs_retrieved} → {docs_after_filter} | "
            f"Relevance: {avg_relevance:.2f} | "
            f"Time: {duration_seconds:.3f}s"
        )
    
    def log_generation(self, test_cases_generated: int, prompt_tokens: int, 
                       completion_tokens: int, duration_seconds: float,
                       had_assumptions: bool = False, asked_clarification: bool = False):
        """Log generation step metrics."""
        self.logger.info(
            f"[GENERATION] Test cases: {test_cases_generated} | "
            f"Tokens: {prompt_tokens}+{completion_tokens} | "
            f"Time: {duration_seconds:.2f}s"
        )
    
    def log_safety_check(self, is_safe: bool, has_evidence: bool, 
                         relevance_score: float, warnings: list):
        """Log safety check results."""
        status = "✓ PASS" if (is_safe and has_evidence) else "⚠ WARN"
        self.logger.info(
            f"[SAFETY] {status} | Relevance: {relevance_score:.2f} | Warnings: {len(warnings)}"
        )
        for w in warnings:
            self.logger.warning(f"  → {w}")

