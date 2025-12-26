"""
Safety Guard - Simple safety checks for RAG pipeline.

This module provides:
1. Minimum document threshold check
2. Basic relevance scoring (keyword matching)  
3. Basic prompt injection detection
"""
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class SafetyResult:
    """Simple container for safety check results."""
    
    def __init__(self):
        self.is_safe = True
        self.has_sufficient_evidence = True
        self.relevance_score = 1.0
        self.injection_detected = False
        self.warnings = []
        self.filtered_docs = []


class SafetyGuard:
    """
    Simple safety guard that checks:
    1. Do we have enough documents?
    2. Are the documents relevant to the query?
    3. Do documents contain prompt injection attempts?
    """
    
    # Dangerous patterns that might be prompt injection
    INJECTION_PATTERNS = [
        r'ignore\s+(all\s+)?(previous|prior)\s+instructions',
        r'disregard\s+(all\s+)?(previous|prior)',
        r'you\s+are\s+now\s+a',
        r'forget\s+everything',
        r'new\s+instructions:',
        r'system\s*:',
    ]
    
    def __init__(
        self,
        min_relevance_score: float = 0.3,
        min_docs_required: int = 1,
        enable_injection_detection: bool = True
    ):
        """
        Args:
            min_relevance_score: Minimum keyword overlap score (0.0 to 1.0)
            min_docs_required: Minimum number of docs needed
            enable_injection_detection: Whether to check for prompt injection
        """
        self.min_relevance_score = min_relevance_score
        self.min_docs_required = min_docs_required
        self.enable_injection_detection = enable_injection_detection
        
        logger.info(f"SafetyGuard initialized: min_relevance={min_relevance_score}, min_docs={min_docs_required}")
    
    def check(self, query, documents) -> SafetyResult:
        """
        Run safety checks on retrieved documents.
        
        Args:
            query: User's question (string)
            documents: List of retrieved Document objects
            
        Returns:
            SafetyResult with check outcomes
        """
        result = SafetyResult()
        
        # Ensure query is a string (fix for the list bug)
        if isinstance(query, list):
            query = query[0] if query else ""
        query = str(query)
        
        # Check 1: Do we have enough documents?
        if len(documents) < self.min_docs_required:
            result.warnings.append(f"Only {len(documents)} documents found (need {self.min_docs_required})")
            result.has_sufficient_evidence = False
        
        # Check 2: Filter by relevance and check for injection
        good_docs = []
        scores = []
        
        for doc in documents:
            content = doc.page_content
            
            # Calculate simple relevance score
            score = self._simple_relevance(query, content)
            scores.append(score)
            
            # Check for prompt injection
            if self.enable_injection_detection and self._has_injection(content):
                result.injection_detected = True
                result.is_safe = False
                result.warnings.append(f"Injection detected in: {doc.metadata.get('source', 'unknown')}")
                # Still include the doc but log the warning
            
            # Keep docs above relevance threshold
            if score >= self.min_relevance_score:
                good_docs.append(doc)
        
        # Calculate average relevance
        result.relevance_score = sum(scores) / len(scores) if scores else 0.0
        result.filtered_docs = good_docs if good_docs else documents  # Fallback to all docs
        
        # Update evidence check
        if len(good_docs) < self.min_docs_required:
            result.has_sufficient_evidence = False
        
        return result
    
    def _simple_relevance(self, query: str, doc_text: str) -> float:
        """
        Simple relevance score based on keyword overlap.
        
        Returns a score between 0.0 and 1.0
        """
        # Extract words (lowercase, only letters)
        query_words = set(re.findall(r'[a-z]+', query.lower()))
        doc_words = set(re.findall(r'[a-z]+', doc_text.lower()))
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'create', 'generate', 'make', 'test', 'case', 'cases', 'use'}
        query_words -= stopwords
        
        if not query_words:
            return 0.5  # Can't calculate, assume medium relevance
        
        # Count how many query words appear in the document
        matches = query_words.intersection(doc_words)
        score = len(matches) / len(query_words)
        
        return score
    
    def _has_injection(self, text: str) -> bool:
        """Check if text contains prompt injection patterns."""
        text_lower = text.lower()
        
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                logger.warning(f"Injection pattern found: {pattern}")
                return True
        
        return False
    
    def get_insufficient_context_response(self, query: str, warnings: List[str]) -> Dict[str, Any]:
        """
        Generate a response when we don't have enough context.
        """
        return {
            "test_cases": [],
            "assumptions_made": [],
            "clarifying_question": (
                f"I don't have enough information to generate test cases for '{query}'. "
                f"Please provide relevant documentation (PRD, specs, screenshots) for this feature."
            )
        }
