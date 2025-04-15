"""Grading interface component for grading LLM responses."""

import time
import os
import sys
from typing import Dict, List, Tuple, Optional, Callable, Any

import streamlit as st
import pandas as pd

# Use absolute imports
from llm_evaler.src.utils import sample_for_grading, truncate_text
from llm_evaler.src.config import MIN_GRADES_REQUIRED, SHOW_ASSERTION_RESULTS_BY_DEFAULT


def grading_interface(
    evaluator,
    on_grade_submitted: Optional[Callable] = None,
) -> Dict[str, bool]:
    """
    Human grading interface component for collecting feedback on LLM outputs.
    
    This interface is designed to efficiently collect human judgments on LLM outputs
    to guide assertion selection, following the EvalGen methodology for aligning
    automated evaluations with human preferences.
    
    Args:
        evaluator: Evaluator instance
        on_grade_submitted: Callback function to call when a grade is submitted
        
    Returns:
        Dictionary of grades (response_id -> bool)
    """
    st.header("Response Grading & Assertion Alignment")
    
    st.markdown("""
    Grade LLM responses as **Good** (👍) or **Bad** (👎) to help select which assertions 
    best align with your preferences. This human feedback calibrates the automated evaluators.
    
    Your grades will be used to:
    - Identify which assertion variants best match your judgments
    - Refine evaluation criteria through iterative feedback
    - Build a transparent report card of aligned assertions
    """)
    
    # Initialize session state variables
    if "graded_responses" not in st.session_state:
        st.session_state.graded_responses = []
    
    if "current_response_id" not in st.session_state:
        # Sample a response to grade
        sampled_ids = sample_for_grading(evaluator, num_samples=1)
        if sampled_ids:
            st.session_state.current_response_id = sampled_ids[0]
        else:
            st.session_state.current_response_id = None
    
    if "sampling_strategy" not in st.session_state:
        st.session_state.sampling_strategy = "alternating"
        
    if "show_explanations" not in st.session_state:
        st.session_state.show_explanations = SHOW_ASSERTION_RESULTS_BY_DEFAULT
    
    if "criteria_drift_notes" not in st.session_state:
        st.session_state.criteria_drift_notes = {}
    
    # Get human grades so far
    human_grades = evaluator.get_grades()
    
    # Calculate stats
    total_responses = len(evaluator.responses)
    graded_responses = len(human_grades)
    remaining_to_grade = max(0, MIN_GRADES_REQUIRED - graded_responses)
    
    # Progress tracker
    st.progress(min(1.0, graded_responses / MIN_GRADES_REQUIRED))
    
    # Stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Responses Graded", graded_responses)
    
    with col2:
        st.metric("Minimum Required", MIN_GRADES_REQUIRED)
    
    with col3:
        st.metric("Remaining", remaining_to_grade)
        
    # Sampling strategy selection
    st.sidebar.subheader("Grading Settings")
    
    sampling_strategy = st.sidebar.radio(
        "Sampling Strategy",
        ["alternating", "random", "low_confidence", "high_confidence"],
        index=["alternating", "random", "low_confidence", "high_confidence"].index(st.session_state.sampling_strategy),
        help="Strategy for sampling responses to grade:\n"
             "- Alternating: Balance between responses with high and low prediction confidence\n"
             "- Random: Completely random sampling\n"
             "- Low Confidence: Focus on responses where assertions disagree\n"
             "- High Confidence: Focus on responses where assertions agree"
    )
    
    if sampling_strategy != st.session_state.sampling_strategy:
        st.session_state.sampling_strategy = sampling_strategy
        st.rerun()
    
    # Option to show assertion explanations
    show_explanations = st.sidebar.checkbox(
        "Show Assertion Results",
        value=st.session_state.show_explanations,
        help="Show the results of assertion executions for each response"
    )
    
    if show_explanations != st.session_state.show_explanations:
        st.session_state.show_explanations = show_explanations
        st.rerun()
    
    if st.sidebar.button("Skip to New Response", help="Skip the current response and grade a different one"):
        # Sample a new response
        sampled_ids = sample_for_grading(
            evaluator, 
            num_samples=1, 
            strategy=st.session_state.sampling_strategy,
            already_graded=st.session_state.graded_responses
        )
        
        if sampled_ids:
            st.session_state.current_response_id = sampled_ids[0]
            st.rerun()
        else:
            st.session_state.current_response_id = None
            st.warning("No more responses to grade!")
            st.rerun()
    
    # I'm tired button
    if st.sidebar.button("I'm tired! 😴", help="Generate alignment report with current grades", type="primary"):
        st.session_state.step = "report_card"
        st.rerun()
    
    # Grading interface
    with st.container():
        st.subheader("Grade This Response")
        
        # Check if we have a current response to grade
        if st.session_state.current_response_id:
            # Find the response
            response = None
            for r in evaluator.responses:
                if r["question_id"] == st.session_state.current_response_id:
                    response = r
                    break
            
            if not response:
                st.error("Response not found!")
                return human_grades
            
            # Show the prompt and question
            with st.expander("View Prompt & Question", expanded=False):
                if "prompt" in response:
                    st.subheader("Prompt")
                    st.code(response["prompt"], language="text")
                
                if "question" in response:
                    st.subheader("Question")
                    st.write(response["question"])
            
            # Show the response
            st.subheader("LLM Response")
            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; background-color: #f9f9f9; color: #000000;">
            {response["response"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Show the correct answer (if available)
            if "answer" in response:
                st.info(f"**Correct Answer:** {response['answer']}")
            
            # Grading buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                good_button = st.button("👍 Good", help="This response meets your quality standards", type="primary", use_container_width=True)
            
            with col2:
                bad_button = st.button("👎 Bad", help="This response does not meet your quality standards", type="secondary", use_container_width=True)
            
            with col3:
                skip_button = st.button("⏩ Skip", help="Skip this response and grade another", use_container_width=True)
            
            # Criteria drift notes
            criteria_note = st.text_area(
                "Observations about criteria or new quality issues (optional)",
                placeholder="Did you notice anything about this response that might suggest new or revised criteria?",
                key="criteria_note"
            )
            
            # Display assertion results if available and show_explanations is enabled
            if show_explanations:
                assertions_to_display = None
                if "assertions" in response:
                    assertions_to_display = response["assertions"]
                elif "evaluation" in response and "assertions" in response["evaluation"]:
                    assertions_to_display = response["evaluation"]["assertions"]
                
                if assertions_to_display:
                    with st.expander("Assertion Results", expanded=False):
                        st.subheader("Current Assertion Results")
                        
                        # Create a temporary dataframe for assertion results display
                        assertion_results = []
                        
                        # Create a more structured view of assertion results by criterion
                        for assertion_id, result in assertions_to_display.items():
                            # Find the associated criterion and variant info
                            criterion_name = "Unknown"
                            variant_info = {}
                            
                            for criterion_id, assertions in evaluator.assertions.items():
                                for assertion in assertions:
                                    if assertion["id"] == assertion_id:
                                        criterion_name = criterion_id
                                        variant_info = assertion
                                        break
                            
                            variant_name = variant_info.get("variant_name", "Unknown")
                            variant_description = variant_info.get("description", "Unknown")
                            
                            assertion_results.append({
                                "Criterion": criterion_name,
                                "Variant": variant_name,
                                "Description": variant_description,
                                "AssertionID": assertion_id,
                                "Passes": "✅" if result["passes"] else "❌",
                                "Error": truncate_text(result["error"]) if result["error"] else ""
                            })
                        
                        if assertion_results:
                            # Group by criterion for better visualization
                            grouped_results = {}
                            for result in assertion_results:
                                criterion = result["Criterion"]
                                if criterion not in grouped_results:
                                    grouped_results[criterion] = []
                                grouped_results[criterion].append(result)
                            
                            # Display results by criterion in tabs
                            criterion_tabs = st.tabs(list(grouped_results.keys()))
                            
                            for criterion_tab, criterion_name in zip(criterion_tabs, grouped_results.keys()):
                                with criterion_tab:
                                    criterion_results = grouped_results[criterion_name]
                                    
                                    # Create columns for each variant
                                    variant_cols = st.columns(len(criterion_results))
                                    
                                    for i, (variant_col, result) in enumerate(zip(variant_cols, criterion_results)):
                                        with variant_col:
                                            st.markdown(f"**{result['Variant']}**")
                                            st.markdown(f"*{result['Description']}*")
                                            st.markdown(f"**Result:** {result['Passes']}")
            
            # Handle form submission
            if good_button or bad_button or skip_button:
                # Record grade if good or bad was clicked
                if good_button or bad_button:
                    grade = good_button  # True for good, False for bad
                    
                    # Add to evaluator
                    evaluator.add_human_grade(st.session_state.current_response_id, grade, criteria_note.strip() if criteria_note.strip() else None)
                    
                    # Store for session state
                    st.session_state.graded_responses.append(st.session_state.current_response_id)
                    
                    # Record criteria note if provided
                    if criteria_note.strip():
                        st.session_state.criteria_drift_notes[st.session_state.current_response_id] = criteria_note.strip()
                    
                    # Call callback if provided
                    if on_grade_submitted:
                        on_grade_submitted(st.session_state.current_response_id, grade)
                
                # Sample next response
                sampled_ids = sample_for_grading(
                    evaluator, 
                    num_samples=1, 
                    strategy=st.session_state.sampling_strategy,
                    already_graded=st.session_state.graded_responses
                )
                
                if sampled_ids:
                    st.session_state.current_response_id = sampled_ids[0]
                    st.rerun()
                else:
                    st.session_state.current_response_id = None
                    st.success("All responses have been graded!")
                    st.rerun()
        
    # Summary statistics and insights
    if len(human_grades) > 5:
        with st.expander("Grading Insights", expanded=False):
            # Show statistics about grades so far
            good_count = sum(1 for grade in human_grades.values() if grade)
            bad_count = sum(1 for grade in human_grades.values() if not grade)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Good Responses", good_count, help="Number of responses marked as good")
            
            with col2:
                st.metric("Bad Responses", bad_count, help="Number of responses marked as bad")
            
            # Display distribution as a bar chart
            st.bar_chart({"Good": [good_count], "Bad": [bad_count]})
    else:
        # No response to grade
        if len(human_grades) >= MIN_GRADES_REQUIRED:
            st.success(f"You've completed {len(human_grades)} grades, which is sufficient for alignment! Continue to the Report Card for results.")
        else:
            st.warning("No responses available for grading.")
    
    return human_grades