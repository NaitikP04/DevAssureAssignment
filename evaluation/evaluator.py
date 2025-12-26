"""
Evaluation Module - Simple tests for RAG output quality.

This checks:
1. Is the output valid JSON?
2. Does it have the required fields (test_cases)?
3. Are the test cases complete?

Run this file directly to test:
    python -m evaluation.evaluator
"""
import json


class RAGEvaluator:
    """
    Simple evaluator for RAG outputs.
    
    Checks if the LLM response is:
    1. Valid JSON
    2. Has required structure
    3. Test cases have necessary fields
    """
    
    def __init__(self):
        self.results = []  # List of (test_name, passed, score)
    
    def evaluate_output(self, output: str, query: str, source_docs: list):
        """
        Evaluate the LLM output quality.
        
        Args:
            output: The LLM's response (should be JSON)
            query: The user's original query
            source_docs: List of source document contents
            
        Returns:
            self (for chaining)
        """
        self.results = []
        
        # Test 1: Is it valid JSON?
        parsed = self._check_json(output)
        
        if parsed:
            # Test 2: Does it have test_cases?
            self._check_structure(parsed)
            
            # Test 3: Are test cases complete?
            self._check_test_cases(parsed)
        
        return self
    
    def _check_json(self, output: str):
        """Check if output is valid JSON."""
        try:
            # Remove markdown code blocks if present
            clean = output.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            if clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            
            parsed = json.loads(clean.strip())
            self.results.append(("json_valid", True, 1.0))
            return parsed
        except:
            self.results.append(("json_valid", False, 0.0))
            return None
    
    def _check_structure(self, parsed: dict):
        """Check if JSON has required structure."""
        if "test_cases" in parsed:
            test_cases = parsed["test_cases"]
            if isinstance(test_cases, list) and len(test_cases) > 0:
                self.results.append(("has_test_cases", True, 1.0))
            elif parsed.get("clarifying_question"):
                # Asking for clarification is also valid
                self.results.append(("has_test_cases", True, 0.8))
            else:
                self.results.append(("has_test_cases", False, 0.0))
        else:
            self.results.append(("has_test_cases", False, 0.0))
    
    def _check_test_cases(self, parsed: dict):
        """Check if test cases have required fields."""
        test_cases = parsed.get("test_cases", [])
        
        if not test_cases:
            return
        
        required_fields = {"id", "title", "steps", "expected_result"}
        scores = []
        
        for tc in test_cases:
            if not isinstance(tc, dict):
                continue
            
            # What percentage of required fields are present?
            present = set(tc.keys()) & required_fields
            score = len(present) / len(required_fields)
            scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        self.results.append(("test_case_quality", avg_score >= 0.5, avg_score))
    
    @property
    def pass_rate(self) -> float:
        """What percentage of tests passed?"""
        if not self.results:
            return 0.0
        passed = sum(1 for _, p, _ in self.results if p)
        return passed / len(self.results)
    
    @property
    def avg_score(self) -> float:
        """Average score across all tests."""
        if not self.results:
            return 0.0
        return sum(s for _, _, s in self.results) / len(self.results)
    
    def print_summary(self):
        """Print a summary of the evaluation."""
        print("\n=== Evaluation Summary ===")
        for name, passed, score in self.results:
            status = "✓" if passed else "✗"
            print(f"  {status} {name}: {score:.2f}")
        print(f"\n  Pass Rate: {self.pass_rate:.1%}")
        print(f"  Avg Score: {self.avg_score:.2f}")


if __name__ == "__main__":
    print("Testing the RAG Evaluator...\n")
    
    # Test 1: Good output
    good_output = json.dumps({
        "test_cases": [
            {
                "id": "TC_001",
                "title": "Login with valid credentials",
                "steps": ["Open login page", "Enter username", "Click login"],
                "expected_result": "User is logged in"
            }
        ],
        "assumptions_made": ["Standard login flow"]
    })
    
    evaluator = RAGEvaluator()
    evaluator.evaluate_output(good_output, "test login", ["login docs"])
    print("Test 1 - Good Output:")
    evaluator.print_summary()
    
    # Test 2: Bad output (not JSON)
    evaluator2 = RAGEvaluator()
    evaluator2.evaluate_output("This is not JSON", "query", [])
    print("\nTest 2 - Invalid JSON:")
    evaluator2.print_summary()
    
    # Test 3: Empty test cases
    empty_output = json.dumps({"test_cases": []})
    evaluator3 = RAGEvaluator()
    evaluator3.evaluate_output(empty_output, "query", [])
    print("\nTest 3 - Empty test cases:")
    evaluator3.print_summary()
    
    print("\n✓ Evaluator tests complete!")

