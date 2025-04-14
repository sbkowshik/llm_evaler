"""Generate assertions for evaluating LLM responses."""

import re
import asyncio
from typing import Dict, List, Optional, Union, Callable, Any, Tuple, TypeVar, Protocol

from llm_evaler.src.llm_client import call_openai_api, call_openai_api_async, evaluate_with_llm, evaluate_with_llm_async


# Define a protocol for assertion functions
class AssertionFunc(Protocol):
    def __call__(self, response: str, question_data: Optional[Dict] = None) -> bool:
        ...


T = TypeVar('T')


def generate_code_assertion(criterion: Dict, example_responses: List[Dict]) -> List[Dict]:
    """
    Generate code-based assertion implementations for a criterion.
    
    Args:
        criterion: Criterion dictionary
        example_responses: List of example responses
        
    Returns:
        List of assertion implementation dictionaries
    """
    # Use asyncio to run the async version
    return asyncio.run(generate_code_assertion_async(criterion, example_responses))


async def generate_code_assertion_async(criterion: Dict, example_responses: List[Dict]) -> List[Dict]:
    """
    Async version of generate_code_assertion.
    
    Args:
        criterion: Criterion dictionary
        example_responses: List of example responses
        
    Returns:
        List of assertion implementation dictionaries
    """
    # Create a prompt for the LLM to generate Python code
    prompt = f"""
    You are an expert Python programmer. I need you to write Python functions to evaluate LLM responses based on the following criterion:
    
    Criterion: {criterion['name']} - {criterion['description']}
    
    The function should:
    1. Take a string parameter 'response' containing the LLM's response
    2. Return a boolean (True if the response passes the criterion, False otherwise)
    3. Be efficient, robust, and handle edge cases
    
    Here are some example responses to consider:
    """
    
    # Add examples
    for i, example in enumerate(example_responses[:3]):
        prompt += f"\nExample {i+1}:\n```\n{example['response']}\n```\n"
    
    prompt += """
    Generate 3 different implementations of this function with different approaches. 
    Each implementation should be complete and standalone.
    
    Use the following format for each implementation:
    
    ```python
    def check_{criterion_name}(response: str) -> bool:
        # Implementation details
        return result
    ```
    
    Make sure each function has proper type hints and docstrings.
    """
    
    # Call the LLM
    result = await call_openai_api_async(prompt, temperature=0.7, max_tokens=2048)
    
    # Parse the code blocks
    code_blocks = extract_code_blocks(result)
    
    # Create assertion implementations
    implementations = []
    
    for i, code in enumerate(code_blocks):
        # Clean up the function name to be valid Python
        clean_code = clean_function_name(code, criterion["name"])
        
        implementations.append({
            "id": f"{criterion['name'].lower().replace(' ', '_')}_code_{i+1}",
            "criterion_id": criterion["name"],
            "type": "code",
            "implementation": clean_code,
            "selectivity": 0.5,  # Initial estimate, will be updated during execution
            "aligned_score": 0.0,  # Will be updated based on human grades
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


async def generate_llm_assertion_async(criterion: Dict, example_responses: List[Dict]) -> List[Dict]:
    """
    Async version of generate_llm_assertion.
    
    Args:
        criterion: Criterion dictionary
        example_responses: List of example responses
        
    Returns:
        List of assertion implementation dictionaries with multiple prompting variants
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
        """,
        
        # Variant 4: Comparative evaluation (context-aware)
        """
        Evaluate this response against the following criterion:
        
        Criterion: {criterion_name} - {criterion_description}
        
        For context, here is the question that the response is addressing:
        Question: {question}
        
        Response to evaluate:
        ```
        {response}
        ```
        
        Consider:
        1. How well does the response address the criterion in the context of this question?
        2. Would a typical user find this response satisfactory for this criterion?
        
        Provide your assessment with either "PASS" or "FAIL".
        """,
        
        # Variant 5: Binary classification with confidence
        """
        System: You are an AI evaluator specialized in assessing responses based on specific criteria.
        
        Criterion: {criterion_name} - {criterion_description}
        
        Response to evaluate:
        ```
        {response}
        ```
        
        Evaluate if this response passes or fails the criterion. Respond with:
        "PASS" if it meets the criterion
        "FAIL" if it does not meet the criterion
        
        Be decisive and respond with only PASS or FAIL.
        """
    ]
    
    # Create assertion implementations
    implementations = []
    
    for i, prompt_template in enumerate(prompts):
        # For the context-aware variant, we need to include question information
        if "Question: {question}" in prompt_template:
            implementation_template = prompt_template.format(
                criterion_name=criterion["name"],
                criterion_description=criterion["description"],
                question="{question}",
                response="{response}"
            )
        else:
            implementation_template = prompt_template.format(
                criterion_name=criterion["name"],
                criterion_description=criterion["description"],
                response="{response}"
            )
        
        implementations.append({
            "id": f"{criterion['name'].lower().replace(' ', '_')}_llm_variant_{i+1}",
            "criterion_id": criterion["name"],
            "type": "llm",
            "variant_name": f"Variant {i+1}",
            "implementation": implementation_template,
            "selectivity": 0.5,  # Initial estimate, will be updated during execution
            "aligned_score": 0.0,  # Will be updated based on human grades
            "description": get_variant_description(i)
        })
    
    return implementations


def get_variant_description(variant_index: int) -> str:
    """
    Get a human-readable description of an assertion variant.
    
    Args:
        variant_index: The index of the variant
        
    Returns:
        A description string
    """
    descriptions = [
        "Direct yes/no evaluation",
        "Step-by-step reasoning approach",
        "Rubric-based assessment",
        "Comparative evaluation (context-aware)",
        "Binary classification with confidence"
    ]
    
    if 0 <= variant_index < len(descriptions):
        return descriptions[variant_index]
    else:
        return f"Variant {variant_index+1}"


def generate_assertions(
    criterion: Dict,
    example_responses: List[Dict],
    use_llm: bool = True,
) -> List[Dict]:
    """
    Generate all assertion implementations for a criterion.
    
    Args:
        criterion: Criterion dictionary
        example_responses: List of example responses
        use_llm: Whether to include LLM-based assertions
        
    Returns:
        List of assertion implementation dictionaries
    """
    # Use asyncio to run the async version
    return asyncio.run(generate_assertions_async(
        criterion=criterion,
        example_responses=example_responses,
        use_llm=use_llm
    ))


async def generate_assertions_async(
    criterion: Dict,
    example_responses: List[Dict],
    use_llm: bool = True,
) -> List[Dict]:
    """
    Async version of generate_assertions.
    
    Args:
        criterion: Criterion dictionary
        example_responses: List of example responses
        use_llm: Whether to include LLM-based assertions
        
    Returns:
        List of assertion implementation dictionaries
    """
    implementations = []
    
    # Create tasks for generating assertions
    tasks = []
    
    if use_llm:
        tasks.append(generate_llm_assertion_async(criterion, example_responses))
    
    # Run the tasks concurrently
    if tasks:
        results = await asyncio.gather(*tasks)
        for result in results:
            implementations.extend(result)
    
    return implementations


def extract_code_blocks(text: str) -> List[str]:
    """
    Extract Python code blocks from a string.
    
    Args:
        text: The string to extract code blocks from
        
    Returns:
        A list of code blocks
    """
    # Pattern for Python code blocks with triple backticks
    pattern = r"```(?:python)?\s*((?:.|\n)*?)```"
    matches = re.finditer(pattern, text)
    
    return [match.group(1).strip() for match in matches]


def clean_function_name(code: str, criterion_name: str) -> str:
    """
    Clean the function name to be valid Python.
    
    Args:
        code: The Python code
        criterion_name: The criterion name
        
    Returns:
        Cleaned Python code
    """
    # Replace spaces and special characters with underscores
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', criterion_name.lower())
    clean_name = re.sub(r'_{2,}', '_', clean_name)  # Replace multiple underscores with single
    
    # Replace function name in code
    pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    match = re.search(pattern, code)
    
    if match:
        old_name = match.group(1)
        new_name = f"check_{clean_name}"
        return code.replace(old_name, new_name)
    
    return code


def execute_code_assertion(
    assertion: Dict,
    response: str
) -> Tuple[bool, Optional[str]]:
    """
    Execute a code-based assertion.
    
    Args:
        assertion: Assertion implementation dictionary
        response: LLM response to evaluate
        
    Returns:
        Tuple of (result, error_message)
    """
    try:
        # Create a namespace for execution
        namespace = {"response": response}
        
        # Execute the assertion code
        exec(assertion["implementation"], globals(), namespace)
        
        # Get the function name from the code
        function_name = re.search(r"def\s+([^(]*)\(", assertion["implementation"]).group(1)
        
        # Call the function
        result = namespace[function_name](response)
        
        return bool(result), None
        
    except Exception as e:
        return False, f"Error executing code assertion: {str(e)}"


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
        question_data: Optional question data for context-aware assertions
        
    Returns:
        Tuple of (result, explanation)
    """
    # Use asyncio to run the async version
    return asyncio.run(execute_llm_assertion_async(assertion, response, question_data))


async def execute_llm_assertion_async(
    assertion: Dict,
    response: str,
    question_data: Optional[Dict] = None
) -> Tuple[bool, Optional[str]]:
    """
    Async version of execute_llm_assertion.
    
    Args:
        assertion: Assertion dictionary
        response: Response to evaluate
        question_data: Optional question data for context-aware assertions
        
    Returns:
        Tuple of (result, explanation)
    """
    # Prepare the prompt
    prompt_template = assertion["implementation"]
    
    # Format the prompt with the response and question data if available
    if question_data and "{question}" in prompt_template:
        prompt = prompt_template.format(response=response, question=question_data.get("question", ""))
    else:
        prompt = prompt_template.format(response=response)
    
    # Call the LLM
    result = await call_openai_api_async(prompt, temperature=0.1, max_tokens=1024)
    
    # Parse the result
    pass_result = "PASS" in result.upper()
    
    # Extract explanation if any
    explanation = result.strip()
    
    return pass_result, explanation


def execute_assertion(
    assertion: Dict,
    response: str,
    question_data: Optional[Dict] = None
) -> Tuple[bool, Optional[str]]:
    """
    Execute any type of assertion.
    
    Args:
        assertion: Assertion dictionary
        response: Response to evaluate
        question_data: Optional question data for context
        
    Returns:
        Tuple of (result, explanation)
    """
    if assertion["type"] == "code":
        return execute_code_assertion(assertion, response)
    elif assertion["type"] == "llm":
        return execute_llm_assertion(assertion, response, question_data)
    else:
        raise ValueError(f"Unknown assertion type: {assertion['type']}") 