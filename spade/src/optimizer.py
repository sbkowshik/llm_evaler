"""Optimizer for selecting the best assertion set in SPADE."""

import math
import itertools
import numpy as np
from typing import Dict, List, Set, Tuple, Optional

from spade.src.config import DEFAULT_FALSE_FAILURE_RATE_THRESHOLD
from llm_evaler.src.alignment import compute_confusion_matrix, compute_alignment_metrics


def compute_selectivity(
    assertion_results: Dict[str, Dict],
) -> float:
    """
    Compute selectivity (pass rate) for an assertion.
    
    Args:
        assertion_results: Dictionary mapping response_id to results
        
    Returns:
        Selectivity as a float between 0.0 and 1.0
    """
    if not assertion_results:
        return 0.0
    
    pass_count = 0
    total = 0
    
    for response_id, result in assertion_results.items():
        if not isinstance(result, dict) or "passes" not in result:
            continue
        
        try:
            if bool(result["passes"]):
                pass_count += 1
            total += 1
        except (ValueError, TypeError):
            continue
    
    return pass_count / total if total > 0 else 0.0


def select_assertions_for_criteria(
    evaluator_results: Dict,
    criteria: List[Dict],
    assertions: List[Dict],
    ffr_threshold: float = DEFAULT_FALSE_FAILURE_RATE_THRESHOLD,
) -> Dict[str, str]:
    """
    Select one assertion for each criterion that meets the FFR threshold.
    In SPADE, we don't use human grades for selection, but rely on selectivity.
    
    Args:
        evaluator_results: Results from evaluating assertions on responses
        criteria: List of criteria
        assertions: List of assertions
        ffr_threshold: Maximum allowed false failure rate
        
    Returns:
        Dictionary mapping criterion_id to selected assertion_id
    """
    # Group assertions by criterion
    assertions_by_criterion = {}
    for assertion in assertions:
        criterion_id = assertion["criterion_id"]
        if criterion_id not in assertions_by_criterion:
            assertions_by_criterion[criterion_id] = []
        assertions_by_criterion[criterion_id].append(assertion)
    
    # Extract assertion results
    assertion_results = {}
    for assertion in assertions:
        assertion_id = assertion["id"]
        results = {}
        
        for response_id, response_result in evaluator_results.items():
            if not isinstance(response_result, dict) or "assertions" not in response_result:
                continue
                
            if assertion_id in response_result["assertions"]:
                results[response_id] = response_result["assertions"][assertion_id]
        
        assertion_results[assertion_id] = results
        
        # Update selectivity
        selectivity = compute_selectivity(results)
        assertion["selectivity"] = selectivity
    
    # Select one assertion per criterion based on selectivity
    # In SPADE, we prefer assertions that don't pass everything (lower selectivity)
    selected_assertions = {}
    
    for criterion_id, criterion_assertions in assertions_by_criterion.items():
        if not criterion_assertions:
            continue
        
        # Sort by selectivity (ascending), preferring assertions that fail more outputs
        sorted_assertions = sorted(criterion_assertions, key=lambda a: a["selectivity"])
        
        # Select the assertion with lowest selectivity that's not too strict
        # If selectivity is too low, this would mean too many false failures
        selected = None
        
        for assertion in sorted_assertions:
            # In SPADE, we don't have human grades to compute FFR directly
            # Instead, we use a heuristic based on selectivity
            # Avoid assertions that fail almost everything (selectivity close to 0)
            # or pass almost everything (selectivity close to 1)
            if 0.1 <= assertion["selectivity"] <= 0.9:
                selected = assertion
                break
        
        # If no good assertion found, take the one with selectivity closest to 0.5
        if not selected and sorted_assertions:
            selected = min(sorted_assertions, key=lambda a: abs(a["selectivity"] - 0.5))
        
        if selected:
            selected_assertions[criterion_id] = selected["id"]
    
    return selected_assertions


def select_minimal_assertion_set(
    evaluator_results: Dict,
    criteria: List[Dict],
    assertions: List[Dict],
    ffr_threshold: float = 0.4,
) -> Dict[str, str]:
    """
    Select minimal set of assertions (one per criterion) optimizing for alignment.
    
    Args:
        evaluator_results: Results from the evaluator
        criteria: List of criteria dictionaries
        assertions: List of assertion dictionaries
        ffr_threshold: Maximum false failure rate allowed
        
    Returns:
        Dictionary mapping criterion_id to selected assertion_id
    """
    # Group assertions by criterion
    assertions_by_criterion = {}
    for assertion in assertions:
        criterion_id = assertion["criterion_id"]
        if criterion_id not in assertions_by_criterion:
            assertions_by_criterion[criterion_id] = []
        assertions_by_criterion[criterion_id].append(assertion)
    
    selected_assertions = {}
    
    # For each criterion, select the assertion with the highest pass rate 
    # that still has FFR below the threshold
    for criterion_id, criterion_assertions in assertions_by_criterion.items():
        best_assertion = None
        best_score = -float('inf')
        
        for assertion in criterion_assertions:
            assertion_id = assertion["id"]
            pass_rate = calculate_assertion_pass_rate(assertion_id, evaluator_results)
            ffr = calculate_assertion_ffr(assertion_id, evaluator_results)
            
            # Only consider assertions with FFR below threshold
            if ffr <= ffr_threshold:
                # Higher pass rate is better
                score = pass_rate
                
                if score > best_score:
                    best_score = score
                    best_assertion = assertion
        
        # Select the best assertion for this criterion
        if best_assertion:
            selected_assertions[criterion_id] = best_assertion["id"]
    
    return selected_assertions


def calculate_assertion_pass_rate(assertion_id: str, evaluator_results: Dict) -> float:
    """
    Calculate the pass rate for a single assertion.
    
    Args:
        assertion_id: ID of the assertion
        evaluator_results: Results from the evaluator
        
    Returns:
        Pass rate as a float between 0 and 1
    """
    total_responses = 0
    passing_responses = 0
    
    for response_id, result in evaluator_results.items():
        if (
            "assertions" in result and 
            assertion_id in result["assertions"] and 
            "passes" in result["assertions"][assertion_id]
        ):
            total_responses += 1
            if result["assertions"][assertion_id]["passes"]:
                passing_responses += 1
    
    if total_responses == 0:
        return 0.0
    
    return passing_responses / total_responses


def calculate_assertion_ffr(assertion_id: str, evaluator_results: Dict) -> float:
    """
    Calculate the false failure rate for a single assertion.
    This is an estimate based on the assumption that most outputs are good.
    
    Args:
        assertion_id: ID of the assertion
        evaluator_results: Results from the evaluator
        
    Returns:
        Estimated FFR as a float between 0 and 1
    """
    # For simplicity, we use a heuristic where we assume that
    # at most 40% of outputs should fail any good assertion
    total_responses = 0
    failing_responses = 0
    
    for response_id, result in evaluator_results.items():
        if (
            "assertions" in result and 
            assertion_id in result["assertions"] and 
            "passes" in result["assertions"][assertion_id]
        ):
            total_responses += 1
            if not result["assertions"][assertion_id]["passes"]:
                failing_responses += 1
    
    if total_responses == 0:
        return 1.0
    
    return failing_responses / total_responses


def compute_spade_metrics(
    evaluator_results: Dict,
    selected_assertions: Dict[str, str],
    responses: List[Dict],
) -> Dict:
    """
    Compute metrics for the selected assertions.
    
    Args:
        evaluator_results: Results from the evaluator
        selected_assertions: Dictionary mapping criterion_id to selected assertion_id
        responses: List of response dictionaries
        
    Returns:
        Dictionary of metrics
    """
    # Calculate overall pass rate (% of responses that pass all selected assertions)
    total_responses = len(responses)
    passing_responses = 0
    
    for response in responses:
        response_id = response["question_id"]
        
        if response_id not in evaluator_results:
            continue
        
        result = evaluator_results[response_id]
        
        # Check if response passes all selected assertions
        passes_all = True
        for criterion_id, assertion_id in selected_assertions.items():
            if (
                "assertions" not in result or
                assertion_id not in result["assertions"] or
                "passes" not in result["assertions"][assertion_id] or
                not result["assertions"][assertion_id]["passes"]
            ):
                passes_all = False
                break
        
        if passes_all:
            passing_responses += 1
    
    overall_pass_rate = passing_responses / total_responses if total_responses > 0 else 0.0
    
    return {
        "overall_pass_rate": overall_pass_rate,
        "total_responses": total_responses,
        "passing_responses": passing_responses,
    }


def compute_alignment_with_human_grades(
    evaluator_results: Dict,
    selected_assertions: Dict[str, str],
    human_grades: Dict[str, bool],
) -> Dict:
    """
    Compute alignment metrics between assertions and human grades.
    
    Args:
        evaluator_results: Results from the evaluator
        selected_assertions: Dictionary mapping criterion_id to selected assertion_id
        human_grades: Dictionary mapping response_id to human grade (True if good)
        
    Returns:
        Dictionary of alignment metrics
    """
    # Build a combined result for all selected assertions
    combined_results = {}
    
    for response_id, result in evaluator_results.items():
        if response_id not in human_grades:
            # Skip responses without human grades
            continue
        
        # A response passes if it passes all selected assertions
        passes_all = True
        for criterion_id, assertion_id in selected_assertions.items():
            if (
                "assertions" not in result or
                assertion_id not in result["assertions"] or
                "passes" not in result["assertions"][assertion_id] or
                not result["assertions"][assertion_id]["passes"]
            ):
                passes_all = False
                break
        
        combined_results[response_id] = {"passes": passes_all}
    
    # Compute confusion matrix
    confusion_matrix = compute_confusion_matrix(combined_results, human_grades)
    
    # Compute alignment metrics
    alignment_metrics = compute_alignment_metrics(confusion_matrix)
    
    # Include confusion matrix in the result
    # Here, confusion_matrix is a tuple (TP, FP, TN, FN), but we store it as a dictionary
    # for easier access in the UI and reporting
    alignment_metrics["confusion_matrix"] = {
        "true_positive": confusion_matrix[0],
        "false_positive": confusion_matrix[1],
        "true_negative": confusion_matrix[2],
        "false_negative": confusion_matrix[3],
    }
    
    return alignment_metrics 