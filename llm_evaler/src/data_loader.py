"""Data loader for the MMLU dataset."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import datasets
import pandas as pd
from tqdm.auto import tqdm

from llm_evaler.src.config import (
    DATASET_NAME,
    SUBSET_NAME,
    SAMPLE_SIZE,
    SAMPLE_FILE,
    RESPONSES_FILE,
)


def load_mmlu_dataset(subset_name: str = SUBSET_NAME) -> datasets.Dataset:
    """
    Load the MMLU dataset from Hugging Face.
    
    Args:
        subset_name: Name of the MMLU subset to load
        
    Returns:
        The loaded dataset
    """
    # Load the dataset
    dataset = datasets.load_dataset(DATASET_NAME, subset_name)
    
    # Return the test split
    return dataset["test"]


def sample_questions(
    dataset: datasets.Dataset, 
    n_samples: int = SAMPLE_SIZE, 
    output_file: Optional[Union[str, Path]] = SAMPLE_FILE,
    seed: int = 42,
    subset_name: str = SUBSET_NAME
) -> List[Dict]:
    """
    Sample a set of questions from the MMLU dataset.
    
    Args:
        dataset: The MMLU dataset
        n_samples: Number of questions to sample
        output_file: Path to save the sampled questions
        seed: Random seed for reproducibility
        subset_name: Name of the dataset subset
        
    Returns:
        List of sampled questions
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Ensure we don't request more samples than available
    n_samples = min(n_samples, len(dataset))
    
    # Sample indices
    indices = random.sample(range(len(dataset)), n_samples)
    
    # Extract the sampled questions
    sampled_questions = []
    for idx in indices:
        question_data = dataset[idx]
        
        # Format options
        options = [
            question_data["choices"][0],
            question_data["choices"][1],
            question_data["choices"][2],
            question_data["choices"][3],
        ]
        
        # Create question object
        question = {
            "id": str(idx),
            "question": question_data["question"],
            "options": options,
            "answer": question_data["answer"],  # This is the index of the correct answer (0-3)
            "subject": subset_name,
        }
        
        sampled_questions.append(question)
    
    # Save to file if specified
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_file, "w") as f:
            json.dump(sampled_questions, f, indent=2)
        
        print(f"Saved {n_samples} sampled questions to {output_file}")
    
    return sampled_questions


def load_sampled_questions(file_path: Union[str, Path] = SAMPLE_FILE) -> List[Dict]:
    """
    Load previously sampled questions from a file.
    
    Args:
        file_path: Path to the saved questions file
        
    Returns:
        List of questions
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Questions file not found: {file_path}")
    
    with open(file_path, "r") as f:
        questions = json.load(f)
    
    return questions


def save_llm_responses(
    responses: List[Dict], 
    output_file: Union[str, Path] = RESPONSES_FILE
) -> None:
    """
    Save LLM responses to a file.
    
    Args:
        responses: List of response objects
        output_file: Path to save the responses
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, "w") as f:
        json.dump(responses, f, indent=2)
    
    print(f"Saved {len(responses)} LLM responses to {output_file}")


def load_llm_responses(file_path: Union[str, Path] = RESPONSES_FILE) -> List[Dict]:
    """
    Load LLM responses from a file.
    
    Args:
        file_path: Path to the saved responses file
        
    Returns:
        List of response objects
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Responses file not found: {file_path}")
    
    with open(file_path, "r") as f:
        responses = json.load(f)
    
    return responses 