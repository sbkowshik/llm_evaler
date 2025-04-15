"""Main script to run SPADE and compare with EvalGen."""

import sys
import os
import json
import argparse
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spade.src.spade import SPADE, run_spade_comparison
from spade.src.config import DEFAULT_FALSE_FAILURE_RATE_THRESHOLD


def main():
    """Run SPADE implementation and comparison with EvalGen."""
    parser = argparse.ArgumentParser(description="Run SPADE evaluation on LLM responses")
    parser.add_argument(
        "--data-file", 
        type=str, 
        default="llm_evaler/data/llm_responses.json",
        help="Path to the file containing LLM responses"
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=50,
        help="Number of responses to evaluate"
    )
    parser.add_argument(
        "--ffr-threshold", 
        type=float, 
        default=DEFAULT_FALSE_FAILURE_RATE_THRESHOLD,
        help="Maximum false failure rate for assertions"
    )
    parser.add_argument(
        "--output-file", 
        type=str, 
        default="spade/results.json",
        help="Path to save results"
    )
    
    args = parser.parse_args()
    
    print("Running SPADE evaluation...")
    print(f"Data file: {args.data_file}")
    print(f"Sample size: {args.sample_size}")
    print(f"FFR threshold: {args.ffr_threshold}")
    
    results = run_spade_comparison(
        data_file=args.data_file,
        sample_size=args.sample_size,
        ffr_threshold=args.ffr_threshold,
    )
    
    print(f"Results saved to {args.output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 