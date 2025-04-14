"""Compute alignment metrics between assertions and human grades."""

from typing import Dict, List, Tuple, Optional

from llm_evaler.src.config import DEFAULT_FALSE_FAILURE_RATE_THRESHOLD


def compute_confusion_matrix(
    assertion_results: Dict[str, Dict],
    human_grades: Dict[str, bool]
) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix for an assertion against human grades.
    
    Args:
        assertion_results: Dictionary mapping response IDs to assertion results
        human_grades: Dictionary mapping response IDs to human grades (True=good, False=bad)
        
    Returns:
        Tuple of (true_positive, false_positive, true_negative, false_negative)
    """
    true_positive = 0   # Assertion passes, human grades as good
    false_positive = 0  # Assertion passes, human grades as bad
    true_negative = 0   # Assertion fails, human grades as bad
    false_negative = 0  # Assertion fails, human grades as good
    
    for response_id, human_grade in human_grades.items():
        if response_id not in assertion_results:
            continue
            
        assertion_result = assertion_results[response_id]["passes"]
        
        if assertion_result and human_grade:
            true_positive += 1
        elif assertion_result and not human_grade:
            false_positive += 1
        elif not assertion_result and not human_grade:
            true_negative += 1
        elif not assertion_result and human_grade:
            false_negative += 1
    
    return true_positive, false_positive, true_negative, false_negative


def compute_alignment_metrics(
    confusion_matrix: Tuple[int, int, int, int]
) -> Dict[str, float]:
    """
    Compute alignment metrics from a confusion matrix.
    
    Args:
        confusion_matrix: Tuple of (true_positive, false_positive, true_negative, false_negative)
        
    Returns:
        Dictionary of metrics
    """
    true_positive, false_positive, true_negative, false_negative = confusion_matrix
    
    # Calculate metrics
    total = true_positive + false_positive + true_negative + false_negative
    if total == 0:
        return {
            "coverage": 0.0,
            "false_failure_rate": 0.0,
            "ffr": 0.0,  # Alias for false_failure_rate
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "accuracy": 0.0,
            "alignment_score": 0.0,
            "alignment": 0.0,  # Alias for alignment_score
        }
    
    # Coverage: Percentage of human-labeled "bad" outputs that are correctly flagged
    total_bad = true_negative + false_positive
    coverage = true_negative / total_bad if total_bad > 0 else 0.0
    
    # False Failure Rate: Percentage of human-labeled "good" outputs that are incorrectly flagged
    total_good = true_positive + false_negative
    ffr = false_negative / total_good if total_good > 0 else 0.0
    
    # Other metrics
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (true_positive + true_negative) / total
    
    # Alignment score: Harmonic mean of coverage and (1 - FFR)
    non_ffr = 1.0 - ffr
    alignment_score = 2 * coverage * non_ffr / (coverage + non_ffr) if (coverage + non_ffr) > 0 else 0.0
    
    return {
        "coverage": coverage,
        "false_failure_rate": ffr,
        "ffr": ffr,  # Alias for false_failure_rate
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "alignment_score": alignment_score,
        "alignment": alignment_score,  # Alias for alignment_score
    }


def compute_assertion_alignment(
    evaluator_results: Dict,
    assertion_id: str,
    human_grades: Dict[str, bool]
) -> Dict:
    """
    Compute alignment metrics for a specific assertion.
    
    Args:
        evaluator_results: Evaluation results from the evaluator
        assertion_id: ID of the assertion to analyze
        human_grades: Dictionary mapping response IDs to human grades
        
    Returns:
        Dictionary of alignment metrics
    """
    # Extract assertion results for all responses
    assertion_results = {}
    for response_id, result in evaluator_results.items():
        if assertion_id in result["assertions"]:
            assertion_results[response_id] = result["assertions"][assertion_id]
    
    # Compute confusion matrix
    confusion_matrix = compute_confusion_matrix(assertion_results, human_grades)
    
    # Compute metrics
    metrics = compute_alignment_metrics(confusion_matrix)
    
    # Add confusion matrix
    metrics["confusion_matrix"] = {
        "true_positive": confusion_matrix[0],
        "false_positive": confusion_matrix[1],
        "true_negative": confusion_matrix[2],
        "false_negative": confusion_matrix[3],
    }
    
    return metrics


def select_best_assertions(
    evaluator,
    ffr_threshold: float = DEFAULT_FALSE_FAILURE_RATE_THRESHOLD,
) -> Dict[str, Dict]:
    """
    Select the best assertion for each criterion based on alignment with human grades.
    
    Args:
        evaluator: Evaluator instance
        ffr_threshold: Maximum acceptable false failure rate
        
    Returns:
        Dictionary mapping criterion IDs to selected assertions
    """
    # Get evaluation results and human grades
    results = evaluator.results
    human_grades = evaluator.get_grades()
    
    if not human_grades:
        raise ValueError("No human grades available. Please grade some responses first.")
    
    selected_assertions = {}
    alignment_metrics = {}
    
    # For each criterion
    for criterion_id, assertion_list in evaluator.assertions.items():
        criterion_assertions = []
        
        # Compute alignment for each assertion
        for assertion in assertion_list:
            assertion_id = assertion["id"]
            metrics = compute_assertion_alignment(results, assertion_id, human_grades)
            
            # Add the metrics to the assertion
            assertion_with_metrics = {**assertion, "metrics": metrics}
            criterion_assertions.append(assertion_with_metrics)
            
            # Store metrics for later use
            alignment_metrics[assertion_id] = metrics
        
        # Filter assertions by FFR threshold
        valid_assertions = [
            a for a in criterion_assertions 
            if a["metrics"]["false_failure_rate"] <= ffr_threshold
        ]
        
        if not valid_assertions:
            # If no assertions meet the threshold, select the one with lowest FFR
            valid_assertions = sorted(
                criterion_assertions,
                key=lambda a: a["metrics"]["false_failure_rate"]
            )
        
        # Select the assertion with highest alignment score
        best_assertion = max(
            valid_assertions,
            key=lambda a: a["metrics"]["alignment_score"]
        )
        
        selected_assertions[criterion_id] = best_assertion
    
    return selected_assertions


def compute_overall_alignment(
    selected_assertions: Dict[str, Dict],
    evaluator_results: Dict,
    human_grades: Dict[str, bool]
) -> Dict:
    """
    Compute overall alignment metrics for a set of selected assertions.
    
    Args:
        selected_assertions: Dictionary mapping criterion IDs to selected assertions
        evaluator_results: Evaluation results from the evaluator
        human_grades: Dictionary mapping response IDs to human grades
        
    Returns:
        Dictionary of overall alignment metrics
    """
    # Create combined assertion result
    # An output passes only if it passes all selected assertions
    combined_results = {}
    
    for response_id in evaluator_results:
        passes_all = True
        
        for criterion_id, assertion in selected_assertions.items():
            assertion_id = assertion["id"]
            
            if assertion_id in evaluator_results[response_id]["assertions"]:
                if not evaluator_results[response_id]["assertions"][assertion_id]["passes"]:
                    passes_all = False
                    break
        
        combined_results[response_id] = {"passes": passes_all}
    
    # Compute confusion matrix
    confusion_matrix = compute_confusion_matrix(combined_results, human_grades)
    
    # Compute metrics
    metrics = compute_alignment_metrics(confusion_matrix)
    
    # Add confusion matrix
    metrics["confusion_matrix"] = {
        "true_positive": confusion_matrix[0],
        "false_positive": confusion_matrix[1],
        "true_negative": confusion_matrix[2],
        "false_negative": confusion_matrix[3],
    }
    
    return metrics 