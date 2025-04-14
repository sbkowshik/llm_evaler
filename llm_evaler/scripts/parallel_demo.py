#!/usr/bin/env python
"""
Demonstration script for parallel LLM processing.

This script shows how to use the parallelized LLM evaluation framework.
"""

import json
import time
import argparse
from typing import Dict, List, Any

from llm_evaler.src.llm_client import generate_responses, configure_concurrent_limit
from llm_evaler.src.assertion_generator import generate_assertions
from llm_evaler.src.criteria_generator import generate_criteria_with_llm
from llm_evaler.src.evaluator import Evaluator


def load_questions(filepath: str, limit: int = None) -> List[Dict]:
    """
    Load questions from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        limit: Maximum number of questions to load
        
    Returns:
        List of question dictionaries
    """
    with open(filepath, 'r') as f:
        questions = json.load(f)
    
    if limit is not None:
        questions = questions[:limit]
    
    return questions


def load_responses(filepath: str, limit: int = None) -> List[Dict]:
    """
    Load existing LLM responses from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        limit: Maximum number of responses to load
        
    Returns:
        List of response dictionaries
    """
    with open(filepath, 'r') as f:
        responses = json.load(f)
    
    if limit is not None:
        responses = responses[:limit]
    
    return responses


def save_responses(responses: List[Dict], filepath: str) -> None:
    """
    Save responses to a JSON file.
    
    Args:
        responses: List of response dictionaries
        filepath: Path to save the JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(responses, f, indent=2)


def main():
    """Main function to demonstrate parallel processing."""
    parser = argparse.ArgumentParser(description="Demonstrate parallel LLM processing")
    parser.add_argument("--questions", type=str, default="llm_evaler/data/mmlu_samples.json",
                        help="Path to questions JSON file")
    parser.add_argument("--responses", type=str, default=None,
                        help="Path to existing responses JSON file (optional)")
    parser.add_argument("--output", type=str, default="llm_evaler/data/parallel_output.json",
                        help="Path to save output JSON file")
    parser.add_argument("--limit", type=int, default=10,
                        help="Limit number of questions/responses to process")
    parser.add_argument("--concurrent", type=int, default=5,
                        help="Maximum number of concurrent API calls")
    args = parser.parse_args()
    
    # Configure concurrent limit for all LLM calls
    configure_concurrent_limit(args.concurrent)
    
    print(f"🚀 Starting parallel LLM processing demo with {args.concurrent} concurrent calls")
    
    # Load questions or responses
    if args.responses:
        print(f"📝 Loading existing responses from {args.responses}")
        responses = load_responses(args.responses, args.limit)
    else:
        print(f"❓ Loading questions from {args.questions}")
        questions = load_questions(args.questions, args.limit)
        
        # Generate responses
        print(f"🤖 Generating responses for {len(questions)} questions")
        start_time = time.time()
        responses = generate_responses(
            questions=questions,
            show_progress=True,
            max_concurrent=args.concurrent
        )
        elapsed = time.time() - start_time
        print(f"✅ Generated {len(responses)} responses in {elapsed:.2f} seconds")
        
        # Save intermediate results
        save_responses(responses, args.output)
        print(f"💾 Saved responses to {args.output}")
    
    # Generate criteria
    print("📋 Generating evaluation criteria")
    start_time = time.time()
    criteria = generate_criteria_with_llm(
        task_description="Evaluate factual accuracy and clarity of MMLU question responses",
        examples=responses[:3],
        num_criteria=3
    )
    elapsed = time.time() - start_time
    print(f"✅ Generated {len(criteria)} criteria in {elapsed:.2f} seconds")
    
    # Generate assertions for each criterion
    print("🔍 Generating assertions for each criterion")
    assertions = {}
    start_time = time.time()
    
    for criterion in criteria:
        print(f"  - Generating assertions for criterion: {criterion['name']}")
        criterion_assertions = generate_assertions(
            criterion=criterion,
            example_responses=responses[:3],
            use_llm=True,
            use_code=True
        )
        assertions[criterion['name']] = criterion_assertions
    
    elapsed = time.time() - start_time
    total_assertions = sum(len(a) for a in assertions.values())
    print(f"✅ Generated {total_assertions} assertions in {elapsed:.2f} seconds")
    
    # Initialize evaluator
    evaluator = Evaluator(responses, assertions)
    
    # Run assertions in parallel
    print("🧪 Running assertions on responses")
    start_time = time.time()
    results = evaluator.run_assertions(
        show_progress=True,
        max_concurrent=args.concurrent
    )
    elapsed = time.time() - start_time
    print(f"✅ Evaluated {len(responses)} responses with {total_assertions} assertions in {elapsed:.2f} seconds")
    
    # Save final results
    results_with_responses = evaluator.get_all_responses_with_results()
    save_responses(results_with_responses, args.output)
    print(f"💾 Saved evaluation results to {args.output}")
    
    # Print statistics
    print("\n📊 Assertion Statistics:")
    for assertion_id, stats in evaluator.get_assertion_stats().items():
        passes = stats["passes"]
        fails = stats["fails"]
        errors = stats["errors"]
        total = passes + fails
        selectivity = stats["selectivity"]
        print(f"  - {assertion_id}: {passes} passes, {fails} fails, {errors} errors, {selectivity:.2f} selectivity")
    
    print("\n🎉 Parallel processing demo completed successfully!")


if __name__ == "__main__":
    main() 