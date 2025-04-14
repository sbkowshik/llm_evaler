"""OpenAI API client for LLM interactions."""

import os
import time
import asyncio
from typing import Dict, List, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor

import openai
from openai import OpenAI
from tqdm.auto import tqdm

from llm_evaler.src.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    EVALUATOR_MODEL,
    PROMPT_TEMPLATE,
)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Semaphore to limit concurrent API calls
# Default is 5 concurrent requests, but can be adjusted based on your API tier
API_SEMAPHORE = asyncio.Semaphore(5)


def format_prompt(question: Dict) -> str:
    """
    Format a question using the prompt template.
    
    Args:
        question: Question dictionary with 'question' and 'options' fields
        
    Returns:
        Formatted prompt string
    """
    return PROMPT_TEMPLATE.format(
        question=question["question"], 
        options=question["options"]
    )


def call_openai_api(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 512,
    retry_limit: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """
    Call the OpenAI API with retry logic.
    
    Args:
        prompt: The prompt to send to the API
        model: The OpenAI model to use
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        retry_limit: Number of retries if the API call fails
        retry_delay: Delay between retries in seconds
        
    Returns:
        Generated text from the API
    """
    # Check API key
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    
    # Retry logic
    for attempt in range(retry_limit):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            if attempt < retry_limit - 1:
                print(f"API call failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                print(f"API call failed after {retry_limit} attempts: {e}")
                raise
    
    # This should never be reached due to the raise in the except block
    return ""


async def call_openai_api_async(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 512,
    retry_limit: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """
    Asynchronous version of call_openai_api with rate limiting.
    
    Args:
        prompt: The prompt to send to the API
        model: The OpenAI model to use
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        retry_limit: Number of retries if the API call fails
        retry_delay: Delay between retries in seconds
        
    Returns:
        Generated text from the API
    """
    # Check API key
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    
    # Use semaphore to limit concurrent API calls
    async with API_SEMAPHORE:
        # Use a thread pool to avoid blocking the event loop with synchronous API calls
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            # Retry logic
            for attempt in range(retry_limit):
                try:
                    response = await loop.run_in_executor(
                        executor,
                        lambda: client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    )
                    
                    return response.choices[0].message.content
                    
                except openai.RateLimitError as e:
                    # Handle rate limits specifically with longer backoff
                    if attempt < retry_limit - 1:
                        backoff = retry_delay * (5 ** attempt)  # Longer exponential backoff for rate limits
                        print(f"Rate limit hit: {e}. Retrying in {backoff} seconds...")
                        await asyncio.sleep(backoff)
                    else:
                        print(f"Rate limit hit after {retry_limit} attempts: {e}")
                        raise
                
                except Exception as e:
                    if attempt < retry_limit - 1:
                        print(f"API call failed: {e}. Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        print(f"API call failed after {retry_limit} attempts: {e}")
                        raise
    
    # This should never be reached due to the raise in the except block
    return ""


def configure_concurrent_limit(limit: int) -> None:
    """
    Configure the limit for concurrent API calls.
    
    Args:
        limit: Maximum number of concurrent API calls
    """
    global API_SEMAPHORE
    API_SEMAPHORE = asyncio.Semaphore(limit)


def generate_responses(
    questions: List[Dict],
    model: str = LLM_MODEL,
    show_progress: bool = True,
    max_concurrent: int = 5,  # Maximum number of concurrent API calls
) -> List[Dict]:
    """
    Generate LLM responses for a list of questions.
    
    Args:
        questions: List of question dictionaries
        model: The OpenAI model to use
        show_progress: Whether to show a progress bar
        max_concurrent: Maximum number of concurrent API calls
        
    Returns:
        List of response objects including the original question and the LLM response
    """
    # Configure the concurrent limit
    configure_concurrent_limit(max_concurrent)
    
    # Use asyncio to parallelize the API calls
    return asyncio.run(generate_responses_async(
        questions=questions,
        model=model,
        show_progress=show_progress,
    ))


async def generate_responses_async(
    questions: List[Dict],
    model: str = LLM_MODEL,
    show_progress: bool = True,
) -> List[Dict]:
    """
    Asynchronous version of generate_responses.
    
    Args:
        questions: List of question dictionaries
        model: The OpenAI model to use
        show_progress: Whether to show a progress bar
        
    Returns:
        List of response objects including the original question and the LLM response
    """
    responses = []
    
    # Set up progress tracking
    total = len(questions)
    pbar = tqdm(total=total, desc="Generating responses") if show_progress else None
    
    # Process questions in batches to control concurrency
    async def process_question(question):
        # Format the prompt
        prompt = format_prompt(question)
        
        # Call the API
        response_text = await call_openai_api_async(prompt, model=model)
        
        # Create response object
        response_obj = {
            "question_id": question["id"],
            "question": question["question"],
            "options": question["options"],
            "ground_truth": question["answer"],
            "response": response_text,
            "grades": [],  # Will store human grades
        }
        
        if pbar:
            pbar.update(1)
            
        return response_obj
    
    # Process all questions concurrently (rate limiting handled by API_SEMAPHORE)
    tasks = [process_question(question) for question in questions]
    all_responses = await asyncio.gather(*tasks)
    responses.extend(all_responses)
    
    if pbar:
        pbar.close()
    
    return responses


async def evaluate_with_llm_async(
    response: str,
    criterion: Dict,
    model: str = EVALUATOR_MODEL,
) -> Dict:
    """
    Asynchronous version of evaluate_with_llm.
    
    Args:
        response: The LLM response to evaluate
        criterion: The evaluation criterion dictionary
        model: The OpenAI model to use for evaluation
        
    Returns:
        Evaluation result with pass/fail status and explanation
    """
    # Format the evaluation prompt
    prompt = f"""
    You are an expert evaluator assessing AI responses to questions.
    
    Criterion: {criterion['name']} - {criterion['description']}
    
    Response to evaluate:
    ```
    {response}
    ```
    
    Does this response satisfy the criterion? Answer with YES or NO, followed by a brief explanation.
    """
    
    # Call the API
    result = await call_openai_api_async(prompt, model=model, temperature=0.3)
    
    # Parse the result (simple heuristic)
    passes = "yes" in result.lower().split()[0]
    
    return {
        "criterion": criterion["name"],
        "passes": passes,
        "explanation": result,
    }


def evaluate_with_llm(
    response: str,
    criterion: Dict,
    model: str = EVALUATOR_MODEL,
) -> Dict:
    """
    Evaluate an LLM response using another LLM as an evaluator.
    
    Args:
        response: The LLM response to evaluate
        criterion: The evaluation criterion dictionary
        model: The OpenAI model to use for evaluation
        
    Returns:
        Evaluation result with pass/fail status and explanation
    """
    # Format the evaluation prompt
    prompt = f"""
    You are an expert evaluator assessing AI responses to questions.
    
    Criterion: {criterion['name']} - {criterion['description']}
    
    Response to evaluate:
    ```
    {response}
    ```
    
    Does this response satisfy the criterion? Answer with YES or NO, followed by a brief explanation.
    """
    
    # Call the API
    result = call_openai_api(prompt, model=model, temperature=0.3)
    
    # Parse the result (simple heuristic)
    passes = "yes" in result.lower().split()[0]
    
    return {
        "criterion": criterion["name"],
        "passes": passes,
        "explanation": result,
    } 