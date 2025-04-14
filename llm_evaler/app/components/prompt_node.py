"""Prompt node component for configuring and testing prompts."""

import time
import os
import sys
from typing import Dict, List, Tuple, Optional, Callable

import streamlit as st

# Use absolute imports
from llm_evaler.src.config import PROMPT_TEMPLATE, SAMPLE_SIZE, SUBSET_NAME
from llm_evaler.src.data_loader import (
    load_mmlu_dataset,
    sample_questions,
    load_sampled_questions,
    save_llm_responses,
    load_llm_responses,
)
from llm_evaler.src.llm_client import generate_responses


def prompt_node(
    on_responses_generated: Optional[Callable] = None
) -> Tuple[List[Dict], str]:
    """
    Prompt node component for configuring and testing prompts.
    
    Args:
        on_responses_generated: Callback function to call when responses are generated
        
    Returns:
        Tuple of (generated_responses, prompt_template)
    """
    st.header("Prompt Configuration")
    
    # Check if we already have generated responses
    try:
        responses = load_llm_responses()
        st.success(f"Loaded {len(responses)} existing LLM responses")
        
        # Create evaluator with loaded responses if it doesn't exist
        if on_responses_generated and st.session_state.evaluator is None:
            on_responses_generated(responses)
        
        # Show regenerate option
        if st.button("Regenerate Responses"):
            st.session_state.regenerate_responses = True
    except FileNotFoundError:
        responses = []
    
    # If we need to generate responses
    if not responses or st.session_state.get("regenerate_responses", False):
        with st.form("prompt_config_form"):
            st.subheader("Prompt Template")
            
            # Prompt template
            prompt_template = st.text_area(
                "Prompt Template", 
                value=PROMPT_TEMPLATE,
                height=200
            )
            
            # Dataset settings
            st.subheader("Dataset Settings")
            
            # Add subset selection
            subset_name = st.selectbox(
                "MMLU Dataset Subset", 
                options=["college_computer_science", "high_school_mathematics", "high_school_physics", "business_ethics"],
                index=0,
                help="Select which MMLU subject area to use"
            )
            
            sample_size = st.slider(
                "Number of questions to sample",
                min_value=10,
                max_value=100,
                value=SAMPLE_SIZE,
                step=10
            )
            
            # Submit button
            submitted = st.form_submit_button("Load Data & Generate Responses")
            
            if submitted:
                with st.spinner("Loading dataset..."):
                    try:
                        # Load dataset
                        st.write("Loading MMLU dataset...")
                        dataset = load_mmlu_dataset(subset_name=subset_name)
                        
                        # Sample questions
                        st.write(f"Sampling {sample_size} questions...")
                        questions = sample_questions(
                            dataset=dataset, 
                            n_samples=sample_size,
                            subset_name=subset_name  # Pass the subset_name to sample_questions
                        )
                        
                        # Generate responses
                        st.write("Generating LLM responses...")
                        responses = generate_responses(questions)
                        
                        # Save responses
                        save_llm_responses(responses)
                        
                        st.success(f"Generated and saved {len(responses)} responses")
                        
                        # Reset regenerate flag
                        st.session_state.regenerate_responses = False
                        
                        # Call callback if provided
                        if on_responses_generated:
                            on_responses_generated(responses)
                            
                    except Exception as e:
                        st.error(f"Error generating responses: {str(e)}")
    
    # Show a sample of the data
    if responses:
        with st.expander("Show Sample Response", expanded=False):
            sample_idx = min(len(responses) - 1, 0)
            sample = responses[sample_idx]
            
            st.subheader("Question")
            st.write(sample["question"])
            
            st.subheader("Options")
            for i, option in enumerate(sample["options"]):
                st.write(f"{chr(65 + i)}) {option}")
            
            st.subheader("LLM Response")
            st.write(sample["response"])
        
        # Add a button to explicitly move to the next step
        if st.button("Continue to Criteria Definition", type="primary") and on_responses_generated:
            on_responses_generated(responses)
    
    return responses, PROMPT_TEMPLATE 