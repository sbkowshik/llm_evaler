"""Configuration settings for the LLM evaluation system."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# MMLU Dataset settings
DATASET_NAME = "cais/mmlu"
SUBSET_NAME = "college_computer_science"
SAMPLE_SIZE = 50  # Number of questions to sample
SAMPLE_FILE = DATA_DIR / "mmlu_samples.json"
RESPONSES_FILE = DATA_DIR / "llm_responses.json"

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"  # Model to use for generating answers
EVALUATOR_MODEL = "gpt-4o-mini"  # Model to use for evaluation

# Prompt template
PROMPT_TEMPLATE = """Answer the following multiple-choice question. Choose the correct option and provide a brief explanation.  
Question: {question}  
Options:  
A) {options[0]}  
B) {options[1]}  
C) {options[2]}  
D) {options[3]}  
"""

# Evaluation settings
DEFAULT_CRITERIA = [
    {
        "name": "Correct Option Selection",
        "description": "Response correctly identifies an option letter (A, B, C, or D)",
        "implementation_type": "llm",
    },
    {
        "name": "Explanation Quality",
        "description": "Explanation is clear, logical, and supports the selected option",
        "implementation_type": "llm",
    },
]

# Alignment settings
DEFAULT_FALSE_FAILURE_RATE_THRESHOLD = 0.15  # 15%
MIN_GRADES_REQUIRED = 5  # Minimum number of grades needed before analyzing alignment 