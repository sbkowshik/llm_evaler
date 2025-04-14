"""Generate evaluation criteria for LLM responses."""

import asyncio
from typing import Dict, List, Optional, Union

from llm_evaler.src.config import DEFAULT_CRITERIA
from llm_evaler.src.llm_client import call_openai_api, call_openai_api_async


def get_default_criteria() -> List[Dict]:
    """
    Get the default evaluation criteria.
    
    Returns:
        List of default criteria dictionaries
    """
    return DEFAULT_CRITERIA.copy()


def generate_criteria_with_llm(
    task_description: str,
    examples: List[Dict],
    num_criteria: int = 5,
    include_defaults: bool = True,
) -> List[Dict]:
    """
    Generate evaluation criteria using an LLM.
    
    Args:
        task_description: Description of the evaluation task
        examples: List of example responses
        num_criteria: Number of criteria to generate
        include_defaults: Whether to include default criteria
        
    Returns:
        List of criteria dictionaries
    """
    # Use asyncio to run the async version
    return asyncio.run(generate_criteria_with_llm_async(
        task_description=task_description,
        examples=examples,
        num_criteria=num_criteria,
        include_defaults=include_defaults
    ))


async def generate_criteria_with_llm_async(
    task_description: str,
    examples: List[Dict],
    num_criteria: int = 5,
    include_defaults: bool = True,
) -> List[Dict]:
    """
    Async version of generate_criteria_with_llm.
    
    Args:
        task_description: Description of the evaluation task
        examples: List of example responses
        num_criteria: Number of criteria to generate
        include_defaults: Whether to include default criteria
        
    Returns:
        List of criteria dictionaries
    """
    # Create the prompt
    prompt = f"""
    You are an expert evaluator of language model responses. I need your help to create evaluation criteria for the following task:
    
    Task: {task_description}
    
    Here are a few example responses to evaluate:
    """
    
    # Add examples
    for i, example in enumerate(examples[:3]):  # Limiting to 3 examples to avoid token limits
        prompt += f"\n\nExample {i+1}:\nQuestion: {example['question']}\n"
        prompt += f"Response: {example['response']}\n"
    
    # Request criteria
    prompt += f"""
    Based on these examples, please suggest {num_criteria} criteria for evaluating responses to this task.
    
    For each criterion, provide:
    1. A short name (2-4 words)
    2. A description of what makes a response pass or fail this criterion
    3. Whether this criterion can be evaluated programmatically (code) or requires human/LLM judgment
    
    Format your response as a bulleted list with these three elements for each criterion.
    """
    
    # Call the LLM
    result = await call_openai_api_async(prompt, temperature=0.7, max_tokens=1024)
    
    # Parse the results (this is a simplistic parser, could be improved)
    parsed_criteria = parse_llm_criteria_output(result)
    
    # Ensure we have exactly num_criteria criteria (or fewer if not enough were generated)
    parsed_criteria = parsed_criteria[:num_criteria]
    
    # Combine with defaults if requested
    if include_defaults:
        return parsed_criteria + get_default_criteria()
    
    return parsed_criteria


def parse_llm_criteria_output(output: str) -> List[Dict]:
    """
    Parse the LLM output into a list of criteria dictionaries.
    
    Args:
        output: The LLM output string
        
    Returns:
        List of criteria dictionaries
    """
    criteria = []
    
    # Split the output into lines
    lines = output.split('\n')
    
    current_criterion = None
    in_description = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Check if this is a new criterion
        if (line.startswith('*') or line.startswith('-') or 
            line.startswith('1.') or line.startswith('2.') or 
            line.startswith('3.') or line.startswith('4.') or 
            line.startswith('5.') or line.startswith('•')):
            
            # Save previous criterion if it exists
            if current_criterion and 'name' in current_criterion and 'description' in current_criterion:
                criteria.append(current_criterion)
            
            # Start a new criterion
            current_criterion = {
                'name': line.lstrip('*- 1234567890.•').strip(),
                'description': '',
                'implementation_type': 'llm',  # Default value
            }
            in_description = False
            
        # If it's a numbered or lettered subpoint under the current criterion
        elif current_criterion and (line.startswith('1.') or line.startswith('2.') or 
                               line.startswith('a.') or line.startswith('b.') or
                               line.startswith('A.') or line.startswith('B.') or
                               line.startswith('•')):
            # This is part of description, not a new criterion
            in_description = True
            current_criterion['description'] += ' ' + line if current_criterion['description'] else line
            
        # If not a new criterion, append to description of current one
        elif current_criterion:
            in_description = True
            
            # Check if this line indicates implementation type
            if 'programmatically' in line.lower() or 'code' in line.lower():
                current_criterion['implementation_type'] = 'code'
            elif 'human' in line.lower() or 'llm' in line.lower() or 'judgment' in line.lower():
                current_criterion['implementation_type'] = 'llm'
            
            # Add to description
            current_criterion['description'] += ' ' + line if current_criterion['description'] else line
    
    # Add the last criterion if it exists
    if current_criterion and 'name' in current_criterion and 'description' in current_criterion:
        criteria.append(current_criterion)
    
    return criteria 