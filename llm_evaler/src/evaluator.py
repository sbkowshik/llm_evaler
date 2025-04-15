"""Run assertions on LLM responses and store results."""

import time
import asyncio
import random
from typing import Dict, List, Optional, Tuple, Any

from tqdm.auto import tqdm

from llm_evaler.src.assertion_generator import execute_assertion, execute_llm_assertion_async
from llm_evaler.src.llm_client import configure_concurrent_limit
from llm_evaler.src.alignment import compute_assertion_alignment


class Evaluator:
    """
    Run assertions on LLM responses and compute metrics.
    """
    
    def __init__(
        self,
        responses: List[Dict],
        assertions: Dict[str, List[Dict]],
    ):
        """
        Initialize the evaluator.
        
        Args:
            responses: List of response objects
            assertions: Dictionary mapping criterion IDs to assertion implementations
        """
        self.responses = responses
        self.assertions = assertions
        self.results = {}  # Will store evaluation results
        self.assertion_stats = {}  # Will store statistics about each assertion
        self.human_grades = {}  # Will store human grades
        self.confidence_scores = {}  # Will store confidence scores for outputs
        
        # Initialize assertion statistics
        for criterion_id, assertion_list in self.assertions.items():
            for assertion in assertion_list:
                assertion_id = assertion["id"]
                self.assertion_stats[assertion_id] = {
                    "passes": 0,
                    "fails": 0,
                    "errors": 0,
                    "selectivity": 0.5,  # Default value, will be updated
                }
        
        # Initialize confidence scores for all responses
        for response in self.responses:
            response_id = response["question_id"]
            self.confidence_scores[response_id] = 0.5  # Default neutral confidence

    def run_assertions(self, show_progress: bool = True, max_concurrent: int = 10) -> Dict:
        """
        Run all assertions on all responses.
        
        Args:
            show_progress: Whether to show a progress bar
            max_concurrent: Maximum number of concurrent LLM assertions
            
        Returns:
            Dictionary of evaluation results
        """
        # Configure the concurrent limit for LLM API calls
        configure_concurrent_limit(max_concurrent)
        
        # Use asyncio to run the async version
        return asyncio.run(self.run_assertions_async(
            show_progress=show_progress,
            max_concurrent=max_concurrent
        ))

    async def run_assertions_async(self, show_progress: bool = True, max_concurrent: int = 10) -> Dict:
        """
        Async version of run_assertions.
        
        Args:
            show_progress: Whether to show a progress bar
            max_concurrent: Maximum number of concurrent LLM assertions
            
        Returns:
            Dictionary of evaluation results
        """
        # Initialize assertion statistics if not already done
        for criterion_id, assertion_list in self.assertions.items():
            for assertion in assertion_list:
                assertion_id = assertion["id"]
                if assertion_id not in self.assertion_stats:
                    self.assertion_stats[assertion_id] = {
                        "passes": 0,
                        "fails": 0,
                        "errors": 0,
                        "selectivity": 0.5,  # Default value, will be updated
                    }
        
        # Count total number of assertions to run
        total_assertions = 0
        for response_obj in self.responses:
            for criterion_id, assertion_list in self.assertions.items():
                total_assertions += len(assertion_list)
        
        # Set up progress bar if requested
        pbar = None
        if show_progress:
            pbar = tqdm(total=total_assertions, desc="Running assertions")
        
        # Create tasks for all response-assertion combinations
        all_tasks = []
        
        # Initialize results for all responses
        for response_obj in self.responses:
            response_id = response_obj["question_id"]
            response_text = response_obj["response"]
            
            # Initialize result structure
            self.results[response_id] = {
                "question_id": response_id,
                "assertions": {},
                "criteria_results": {},
                "overall_pass": True,  # Will be updated if any assertion fails
            }
            
            # Initialize criteria results
            for criterion_id in self.assertions:
                self.results[response_id]["criteria_results"][criterion_id] = {
                    "pass": True,  # Will be updated if any assertion fails
                    "details": []
                }
            
            # Create tasks for all assertions for this response
            for criterion_id, assertion_list in self.assertions.items():
                for assertion in assertion_list:
                    # Create task based on assertion type
                    if assertion["type"] == "llm":
                        # For LLM assertions, use the async version
                        task = self._run_llm_assertion(assertion, response_text, response_id, criterion_id)
                    else:
                        # For code assertions, use a wrapper to run in an executor
                        task = self._run_code_assertion(assertion, response_text, response_id, criterion_id)
                    
                    all_tasks.append(task)
        
        # Run all tasks concurrently (API rate limiting is handled by semaphore)
        # This processes all responses and all assertions in parallel
        for i in range(0, len(all_tasks), max_concurrent * 2):  # Process in larger chunks for efficiency
            chunk = all_tasks[i:i+max_concurrent * 2]
            await asyncio.gather(*chunk)
            
            # Update progress bar
            if pbar:
                pbar.update(len(chunk))
        
        # Close progress bar if requested
        if show_progress and pbar:
            pbar.close()
        
        # Update selectivity for each assertion
        for assertion_id, stats in self.assertion_stats.items():
            total = stats["passes"] + stats["fails"]
            stats["selectivity"] = stats["passes"] / total if total > 0 else 0.5
            
            # Also update the assertion object
            for criterion_assertions in self.assertions.values():
                for assertion in criterion_assertions:
                    if assertion["id"] == assertion_id:
                        assertion["selectivity"] = stats["selectivity"]
        
        # Update confidence scores after running all assertions
        self._update_confidence_scores()
        
        return self.results
    
    async def _run_llm_assertion(self, assertion, response_text, response_id, criterion_id):
        """
        Run an LLM assertion asynchronously.
        
        Args:
            assertion: Assertion to run
            response_text: Text of the response
            response_id: ID of the response
            criterion_id: ID of the criterion
        """
        try:
            # Run the assertion
            passes, error_msg = await execute_llm_assertion_async(assertion, response_text)
            
            # Store and update results
            await self._update_assertion_result(
                assertion=assertion,
                response_id=response_id,
                criterion_id=criterion_id,
                passes=passes,
                error_msg=error_msg
            )
            
        except Exception as e:
            # Handle any unexpected errors
            await self._update_assertion_result(
                assertion=assertion,
                response_id=response_id,
                criterion_id=criterion_id,
                passes=False,
                error_msg=f"Unexpected error: {str(e)}"
            )
    
    async def _run_code_assertion(self, assertion, response_text, response_id, criterion_id):
        """
        Run a code assertion in a thread pool.
        
        Args:
            assertion: Assertion to run
            response_text: Text of the response
            response_id: ID of the response
            criterion_id: ID of the criterion
        """
        # Run the code assertion in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            passes, error_msg = await loop.run_in_executor(
                None, 
                lambda: execute_assertion(assertion, response_text)
            )
            
            # Store and update results
            await self._update_assertion_result(
                assertion=assertion,
                response_id=response_id,
                criterion_id=criterion_id,
                passes=passes,
                error_msg=error_msg
            )
            
        except Exception as e:
            # Handle any unexpected errors
            await self._update_assertion_result(
                assertion=assertion,
                response_id=response_id,
                criterion_id=criterion_id,
                passes=False,
                error_msg=f"Unexpected error: {str(e)}"
            )
    
    async def _update_assertion_result(self, assertion, response_id, criterion_id, passes, error_msg):
        """
        Update results with an assertion result.
        
        Args:
            assertion: Assertion that was run
            response_id: ID of the response
            criterion_id: ID of the criterion
            passes: Whether the assertion passed
            error_msg: Error message, if any
        """
        assertion_id = assertion["id"]
        
        # Store the result
        self.results[response_id]["assertions"][assertion_id] = {
            "passes": passes,
            "error": error_msg,
        }
        
        # Add to criteria results
        self.results[response_id]["criteria_results"][criterion_id]["details"].append({
            "assertion_id": assertion_id,
            "passes": passes,
            "error": error_msg,
        })
        
        # Update criteria pass status
        if not passes:
            self.results[response_id]["criteria_results"][criterion_id]["pass"] = False
            self.results[response_id]["overall_pass"] = False
        
        # Update assertion statistics
        if error_msg:
            self.assertion_stats[assertion_id]["errors"] += 1
        elif passes:
            self.assertion_stats[assertion_id]["passes"] += 1
        else:
            self.assertion_stats[assertion_id]["fails"] += 1

    def add_human_grade(self, response_id: str, grade: bool, comment: Optional[str] = None) -> None:
        """
        Add a human grade for a response.
        
        Args:
            response_id: ID of the response
            grade: True for good, False for bad
            comment: Optional comment on the grade
        """
        # Find the response
        for i, response in enumerate(self.responses):
            if response["question_id"] == response_id:
                # Add the grade
                grade_obj = {
                    "grade": grade,
                    "timestamp": time.time(),
                }
                
                if comment:
                    grade_obj["comment"] = comment
                
                self.responses[i]["grades"] = self.responses[i].get("grades", []) + [grade_obj]
                self.human_grades[response_id] = grade
                
                # Update confidence scores after adding a new grade
                self._update_confidence_scores()
                return
        
        # If we get here, the response ID was not found
        raise ValueError(f"Response ID not found: {response_id}")
    
    def get_grades(self) -> Dict[str, bool]:
        """
        Get all human grades.
        
        Returns:
            Dictionary mapping response IDs to grades
        """
        return self.human_grades.copy()
    
    def get_assertion_stats(self) -> Dict:
        """
        Get statistics for all assertions.
        
        Returns:
            Dictionary of assertion statistics
        """
        return self.assertion_stats.copy()
    
    def get_response_with_results(self, response_id: str) -> Optional[Dict]:
        """
        Get a response with its evaluation results.
        
        Args:
            response_id: ID of the response
            
        Returns:
            Response object with results, or None if not found
        """
        # Find the response
        for response in self.responses:
            if response["question_id"] == response_id:
                # Add the results
                result = {**response}
                if response_id in self.results:
                    result["evaluation"] = self.results[response_id]
                return result
        
        return None
    
    def get_all_responses_with_results(self) -> List[Dict]:
        """
        Get all responses with their evaluation results.
        
        Returns:
            List of response objects with results
        """
        result = []
        
        for response in self.responses:
            response_id = response["question_id"]
            response_with_results = {**response}
            
            if response_id in self.results:
                response_with_results["evaluation"] = self.results[response_id]
            
            result.append(response_with_results)
        
        return result
    
    def compute_alignment_metrics(self, force_recompute: bool = False) -> Dict[str, Dict]:
        """
        Compute alignment metrics for all assertions against human grades.
        
        This measures how well each assertion aligns with human judgment.
        
        Args:
            force_recompute: Whether to force recomputing metrics even if cached
            
        Returns:
            Dictionary mapping criterion IDs to alignment metrics
        """
        # Check if we have human grades
        human_grades = self.get_grades()
        if not human_grades:
            return {}
            
        # Initialize metrics dictionary
        metrics_by_criterion = {}
        
        # For each criterion
        for criterion_id, assertion_list in self.assertions.items():
            criterion_metrics = {
                "coverage": 0.0,
                "ffr": 0.0,  # False Failure Rate
                "alignment": 0.0,
                "variant_scores": {}
            }
            
            # Compute metrics for each assertion variant
            for assertion in assertion_list:
                assertion_id = assertion["id"]
                metrics = compute_assertion_alignment(self.results, assertion_id, human_grades)
                
                # Store the alignment score for this variant
                criterion_metrics["variant_scores"][assertion_id] = metrics["alignment_score"]
                
                # Update coverage and FFR if this is the best variant so far
                if metrics["alignment_score"] > criterion_metrics["alignment"]:
                    criterion_metrics["coverage"] = metrics["coverage"]
                    criterion_metrics["ffr"] = metrics["ffr"]
                    criterion_metrics["alignment"] = metrics["alignment"]
            
            metrics_by_criterion[criterion_id] = criterion_metrics
        
        return metrics_by_criterion
    
    def _update_confidence_scores(self) -> None:
        """
        Update confidence scores for all responses based on assertion results and selectivity.
        
        This implements the EvalGen approach where confidence scores estimate the likelihood
        that an LLM output is of low quality, based on assertion results and selectivity.
        """
        # Skip if we don't have any assertion results yet
        if not self.results:
            return
            
        # For each response
        for response_id in self.results:
            # Skip responses that have already been graded by a human
            if response_id in self.human_grades:
                continue
                
            # Reset confidence score
            confidence = 0.5  # Default neutral confidence
            
            # Count assertions that failed this response
            failed_assertions = []
            
            # For each assertion
            for criterion_id, assertion_list in self.assertions.items():
                for assertion in assertion_list:
                    assertion_id = assertion["id"]
                    
                    # Skip if this assertion hasn't been run on this response
                    if assertion_id not in self.results[response_id]["assertions"]:
                        continue
                        
                    # Get assertion result
                    result = self.results[response_id]["assertions"][assertion_id]
                    
                    # Skip assertions with errors
                    if result.get("error"):
                        continue
                        
                    # If assertion failed, add it to the list
                    if not result["passes"]:
                        failed_assertions.append(assertion_id)
            
            # Update confidence score based on failed assertions and their selectivity
            if failed_assertions:
                # For each failed assertion, adjust confidence based on selectivity
                for assertion_id in failed_assertions:
                    selectivity = self.assertion_stats[assertion_id]["selectivity"]
                    # The lower the selectivity (fewer passes), the more significant a failure is
                    confidence += (1 - selectivity) / len(failed_assertions)
                
                # Cap confidence at 1.0
                confidence = min(confidence, 1.0)
            
            # Store updated confidence score
            self.confidence_scores[response_id] = confidence
    
    def sample_response_for_grading(self) -> Optional[str]:
        """
        Sample an LLM output for the user to grade based on confidence scores.
        
        This implements the Grading Sampler component of EvalGen, which samples
        LLM outputs for the user to give binary feedback on.
        
        Returns:
            ID of the response to grade, or None if all responses have been graded
        """
        # Get ungraded responses
        ungraded_ids = [
            response_id for response_id in self.results
            if response_id not in self.human_grades
        ]
        
        # If all responses have been graded, return None
        if not ungraded_ids:
            return None
        
        # Use confidence scores to bias sampling toward potentially poor outputs
        if self.confidence_scores:
            # Sort ungraded responses by confidence score (highest first)
            sorted_ids = sorted(
                ungraded_ids,
                key=lambda id: self.confidence_scores.get(id, 0.5),
                reverse=True
            )
            
            # Implement a sampling strategy biased toward high confidence scores
            # but with some exploration of lower-confidence responses
            
            # 70% of the time, pick from the top 30% of responses
            if random.random() < 0.7:
                top_n = max(1, int(len(sorted_ids) * 0.3))
                return random.choice(sorted_ids[:top_n])
            # 30% of the time, pick from all ungraded responses
            else:
                return random.choice(ungraded_ids)
        else:
            # If no confidence scores, pick randomly
            return random.choice(ungraded_ids)
    
    def get_next_response_to_grade(self) -> Optional[Dict]:
        """
        Get the next response the user should grade, with its details.
        
        This implements the Grading Sampler component of EvalGen, providing
        the next LLM output for the user to grade.
        
        Returns:
            Response object to grade, or None if all responses have been graded
        """
        # Sample a response ID
        response_id = self.sample_response_for_grading()
        
        # If no ungraded responses, return None
        if response_id is None:
            return None
        
        # Get the response details
        return self.get_response_with_results(response_id)
    
    def get_confidence_scores(self) -> Dict[str, float]:
        """
        Get confidence scores for all responses.
        
        Returns:
            Dictionary mapping response IDs to confidence scores
        """
        return self.confidence_scores.copy() 