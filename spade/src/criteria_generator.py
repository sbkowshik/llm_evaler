"""Generate evaluation criteria automatically for SPADE."""

import asyncio
from typing import Dict, List, Optional, Union

from spade.src.config import DEFAULT_SPADE_CRITERIA
from llm_evaler.src.llm_client import call_openai_api, call_openai_api_async


def get_default_criteria() -> List[Dict]:
    """
    Get the default evaluation criteria for SPADE.
    
    Returns:
        List of default criteria dictionaries
    """
    return DEFAULT_SPADE_CRITERIA.copy()


def generate_criteria_with_llm(
    prompt: str,
    example_outputs: List[Dict],
    num_criteria: int = 5,
    include_defaults: bool = True,
) -> List[Dict]:
    """
    Generate evaluation criteria automatically using an LLM without human input.
    
    Args:
        prompt: The input prompt used to generate the LLM outputs
        example_outputs: List of example LLM outputs
        num_criteria: Number of criteria to generate
        include_defaults: Whether to include default criteria
        
    Returns:
        List of criteria dictionaries
    """
    # Use asyncio to run the async version
    return asyncio.run(generate_criteria_with_llm_async(
        prompt=prompt,
        example_outputs=example_outputs,
        num_criteria=num_criteria,
        include_defaults=include_defaults
    ))


async def generate_criteria_with_llm_async(
    prompt: str,
    example_outputs: List[Dict],
    num_criteria: int = 5,
    include_defaults: bool = True,
) -> List[Dict]:
    """
    Async version of generate_criteria_with_llm.
    
    Args:
        prompt: The input prompt used to generate the LLM outputs
        example_outputs: List of example LLM outputs
        num_criteria: Number of criteria to generate
        include_defaults: Whether to include default criteria
        
    Returns:
        List of criteria dictionaries
    """
    # Create the prompt for the SPADE criteria generation
    spade_prompt = f"""
    You are an expert evaluator of language model responses. I need your help to automatically create evaluation criteria 
    for a language model pipeline. 
    
    The following is the prompt template used to generate responses:
    
    ```
    {prompt}
    ```
    
    Here are {min(5, len(example_outputs))} example outputs from this prompt:
    """
    
    # Add examples
    for i, example in enumerate(example_outputs[:5]):
        spade_prompt += f"\n\nExample {i+1}:\n"
        if "question" in example:
            spade_prompt += f"Question: {example['question']}\n"
        spade_prompt += f"Response: {example['response']}\n"
    
    # Request criteria in SPADE style (focusing on pass/fail binary checks)
    spade_prompt += f"""
    Based on the prompt template and these example outputs, suggest {num_criteria} binary evaluation criteria 
    that can be used to automatically determine if an output is good or bad. 
    
    For each criterion:
    1. Provide a short, descriptive name (2-4 words)
    2. Give a clear description of what makes a response pass or fail this criterion
    3. Each criterion should be binary (pass/fail) and ideally objective
    
    These criteria will be used to automatically evaluate all future outputs from this prompt template 
    without human intervention. Focus on important aspects that make a good response.
    
    Format your response as a bulleted list with the name and description for each criterion.
    """
    
    # Call the LLM
    result = await call_openai_api_async(spade_prompt, temperature=0.7, max_tokens=1024)
    
    # Parse the results
    parsed_criteria = parse_llm_criteria_output(result)
    
    # Ensure we have at most num_criteria criteria
    parsed_criteria = parsed_criteria[:num_criteria]
    
    # For SPADE, all criteria use LLM as implementation type
    for criterion in parsed_criteria:
        criterion["implementation_type"] = "llm"
    
    # Combine with defaults if requested
    if include_defaults:
        # Add only default criteria that don't overlap with generated ones
        default_criteria = get_default_criteria()
        default_names = {c["name"].lower() for c in default_criteria}
        generated_names = {c["name"].lower() for c in parsed_criteria}
        
        for criterion in default_criteria:
            if criterion["name"].lower() not in generated_names:
                parsed_criteria.append(criterion)
    
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
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Check if this is a new criterion (bullet point or numbered)
        if (line.startswith('*') or line.startswith('-') or 
            line.startswith('1.') or line.startswith('2.') or 
            line.startswith('3.') or line.startswith('4.') or 
            line.startswith('5.') or line.startswith('•')):
            
            # Save previous criterion if it exists
            if current_criterion and 'name' in current_criterion and 'description' in current_criterion:
                criteria.append(current_criterion)
            
            # Start a new criterion with default implementation type of "llm"
            current_criterion = {
                'name': line.lstrip('*- 1234567890.•').strip(),
                'description': '',
                'implementation_type': 'llm',
            }
            
        # If not a new criterion, append to description of current one
        elif current_criterion:
            # Add to description
            current_criterion['description'] += ' ' + line if current_criterion['description'] else line
    
    # Add the last criterion if it exists
    if current_criterion and 'name' in current_criterion and 'description' in current_criterion:
        criteria.append(current_criterion)
    
    return criteria 