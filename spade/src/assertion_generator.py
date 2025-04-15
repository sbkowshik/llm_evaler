"""Generate assertions for evaluating LLM responses in SPADE."""

import asyncio
from typing import Dict, List, Optional, Any, Tuple

from llm_evaler.src.llm_client import call_openai_api_async, evaluate_with_llm_async


async def generate_llm_assertion_async(criterion: Dict, example_responses: List[Dict]) -> List[Dict]:
    """
    Async function to generate LLM-based assertions for a criterion.
    
    Args:
        criterion: Criterion dictionary
        example_responses: List of example responses
        
    Returns:
        List of assertion implementation dictionaries
    """
    # Define multiple prompt templates with different prompting strategies
    prompts = [
        # Variant 1: Direct yes/no evaluation
        """
        You are an evaluator assessing responses based on this criterion:
        
        Criterion: {criterion_name} - {criterion_description}
        
        Response to evaluate:
        ```
        {response}
        ```
        
        Does this response PASS or FAIL the criterion? Answer with PASS or FAIL only.
        """,
        
        # Variant 2: Step-by-step reasoning approach
        """
        You are an evaluator assessing responses based on this criterion:
        
        Criterion: {criterion_name} - {criterion_description}
        
        Response to evaluate:
        ```
        {response}
        ```
        
        Follow these steps:
        1. Identify the key aspects of the response relevant to the criterion
        2. Analyze how well each aspect satisfies the criterion
        3. Make a final judgment
        
        Conclude with either "PASS" or "FAIL".
        """,
        
        # Variant 3: Rubric-based assessment
        """
        As an expert evaluator, assess the following response based on this criterion:
        
        Criterion: {criterion_name} - {criterion_description}
        
        Use this rubric:
        - Excellent (PASS): Response fully satisfies all aspects of the criterion
        - Good (PASS): Response satisfies most aspects with minor issues
        - Fair (FAIL): Response partially satisfies the criterion but has significant issues
        - Poor (FAIL): Response fails to satisfy the criterion
        
        Response:
        ```
        {response}
        ```
        
        Rate the response according to the rubric and explain your reasoning.
        End with either "PASS" or "FAIL".
        """
    ]
    
    # Generate assertions for each prompt variant
    implementations = []
    
    for i, prompt_template in enumerate(prompts):
        variant_description = get_variant_description(i)
        
        implementations.append({
            "id": f"{criterion['name'].lower().replace(' ', '_')}_llm_{i+1}",
            "criterion_id": criterion["name"],
            "type": "llm",
            "implementation": prompt_template.format(
                criterion_name=criterion["name"],
                criterion_description=criterion["description"],
                response="{response}",  # This will be filled in when evaluating
                question="{question}"   # This will be filled in when evaluating
            ),
            "selectivity": 0.5,  # Initial estimate, will be updated during execution
            "aligned_score": 0.0,  # Will be updated during optimization
            "variant": variant_description
        })
    
    return implementations


def generate_llm_assertion(criterion: Dict, example_responses: List[Dict]) -> List[Dict]:
    """
    Generate LLM-based assertion implementations for a criterion.
    
    Args:
        criterion: Criterion dictionary
        example_responses: List of example responses
        
    Returns:
        List of assertion implementation dictionaries
    """
    # Use asyncio to run the async version
    return asyncio.run(generate_llm_assertion_async(criterion, example_responses))


def get_variant_description(variant_index: int) -> str:
    """
    Get a description for a prompt variant.
    
    Args:
        variant_index: Index of the variant
        
    Returns:
        Description of the variant
    """
    descriptions = [
        "Direct PASS/FAIL evaluation",
        "Step-by-step reasoning",
        "Rubric-based assessment",
    ]
    
    if 0 <= variant_index < len(descriptions):
        return descriptions[variant_index]
    else:
        return f"Variant {variant_index + 1}"


async def execute_llm_assertion_async(
    assertion: Dict,
    response: str,
    question_data: Optional[Dict] = None
) -> Tuple[bool, Optional[str]]:
    """
    Execute an LLM-based assertion asynchronously.
    
    Args:
        assertion: Assertion dictionary
        response: Response to evaluate
        question_data: Optional question data for context
        
    Returns:
        Tuple of (passes, explanation)
    """
    # Format the prompt with the response and question
    prompt_template = assertion["implementation"]
    
    # Replace placeholders
    prompt = prompt_template.replace("{response}", response)
    
    # Include question if available
    if question_data and "{question}" in prompt:
        question = question_data.get("question", "")
        prompt = prompt.replace("{question}", question)
    
    # Evaluate with LLM
    eval_result = await evaluate_with_llm_async(prompt)
    
    # Extract pass/fail result
    passes = "PASS" in eval_result.upper() and not ("FAIL" in eval_result.upper())
    
    return passes, eval_result


def execute_llm_assertion(
    assertion: Dict,
    response: str,
    question_data: Optional[Dict] = None
) -> Tuple[bool, Optional[str]]:
    """
    Execute an LLM-based assertion.
    
    Args:
        assertion: Assertion dictionary
        response: Response to evaluate
        question_data: Optional question data for context
        
    Returns:
        Tuple of (passes, explanation)
    """
    # Use asyncio to run the async version
    return asyncio.run(execute_llm_assertion_async(assertion, response, question_data))


async def generate_assertions_async(
    criterion: Dict,
    example_responses: List[Dict],
) -> List[Dict]:
    """
    Generate assertions for a criterion asynchronously.
    In SPADE, we only use LLM-based assertions.
    
    Args:
        criterion: Criterion dictionary
        example_responses: List of example responses
        
    Returns:
        List of assertion dictionaries
    """
    return await generate_llm_assertion_async(criterion, example_responses)


def generate_assertions(
    criterion: Dict,
    example_responses: List[Dict],
) -> List[Dict]:
    """
    Generate assertions for a criterion.
    In SPADE, we only use LLM-based assertions.
    
    Args:
        criterion: Criterion dictionary
        example_responses: List of example responses
        
    Returns:
        List of assertion dictionaries
    """
    return asyncio.run(generate_assertions_async(criterion, example_responses)) 