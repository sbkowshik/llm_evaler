"""Utility functions for the LLM evaluation system."""

import os
import random
from typing import Dict, List, Optional, Union, Any, Tuple

from llm_evaler.src.config import MIN_GRADES_REQUIRED


def sample_for_grading(
    evaluator,
    num_samples: int = 1,
    strategy: str = "alternating",
    already_graded: List[str] = None,
) -> List[str]:
    """
    Sample responses for human grading.
    
    Args:
        evaluator: Evaluator instance
        num_samples: Number of responses to sample
        strategy: Sampling strategy ("alternating", "random", "low_confidence", "high_confidence")
        already_graded: List of response IDs that have already been graded
        
    Returns:
        List of response IDs to grade
    """
    already_graded = already_graded or []
    
    # Get all responses that haven't been graded yet
    ungraded_ids = []
    for response in evaluator.responses:
        response_id = response["question_id"]
        if response_id not in already_graded:
            ungraded_ids.append(response_id)
    
    # If no ungraded responses left, return empty list
    if not ungraded_ids:
        return []
    
    # Make sure we don't request more samples than available
    num_samples = min(num_samples, len(ungraded_ids))
    
    # If we don't have enough grades yet, or no assertions have been run,
    # just sample randomly to bootstrap the process
    human_grades = evaluator.get_grades()
    if len(human_grades) < MIN_GRADES_REQUIRED or not evaluator.results:
        return random.sample(ungraded_ids, num_samples)
    
    # If using the alternating strategy
    if strategy == "alternating":
        # Alternate between high and low confidence
        if len(human_grades) % 2 == 0:
            return sample_by_confidence(evaluator, num_samples, ungraded_ids, "low")
        else:
            return sample_by_confidence(evaluator, num_samples, ungraded_ids, "high")
    
    # Other strategies
    elif strategy == "random":
        return random.sample(ungraded_ids, num_samples)
    elif strategy in ["low_confidence", "high_confidence"]:
        confidence_type = "low" if strategy == "low_confidence" else "high"
        return sample_by_confidence(evaluator, num_samples, ungraded_ids, confidence_type)
    else:
        # Default to random if unknown strategy
        return random.sample(ungraded_ids, num_samples)


def sample_by_confidence(
    evaluator,
    num_samples: int,
    response_ids: List[str],
    confidence_type: str,
) -> List[str]:
    """
    Sample responses based on confidence scores.
    
    Args:
        evaluator: Evaluator instance
        num_samples: Number of responses to sample
        response_ids: List of candidate response IDs
        confidence_type: "high" or "low" confidence
        
    Returns:
        List of sampled response IDs
    """
    # Calculate confidence scores
    confidence_scores = {}
    
    for response_id in response_ids:
        # Count how many assertions the response passes
        if response_id not in evaluator.results:
            # Default to 0.5 if no results yet
            confidence_scores[response_id] = 0.5
            continue
            
        result = evaluator.results[response_id]
        passes = 0
        total = 0
        
        for assertion_result in result["assertions"].values():
            if assertion_result["error"] is None:  # Only count assertions without errors
                total += 1
                if assertion_result["passes"]:
                    passes += 1
        
        # Confidence score is the proportion of assertions passed
        confidence_scores[response_id] = passes / total if total > 0 else 0.5
    
    # Sort by confidence score
    if confidence_type == "high":
        sorted_ids = sorted(response_ids, key=lambda x: confidence_scores[x], reverse=True)
    else:  # low confidence
        sorted_ids = sorted(response_ids, key=lambda x: confidence_scores[x])
    
    # Return the top N
    return sorted_ids[:num_samples]


def format_percentage(value: float) -> str:
    """
    Format a value as a percentage string.
    
    Args:
        value: Value to format (0.0 to 1.0)
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.1f}%"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length, adding ellipsis if truncated.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - 3] + "..."


def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True) 