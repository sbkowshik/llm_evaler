"""Grading interface component for grading LLM responses."""

import time
import os
import sys
from typing import Dict, List, Tuple, Optional, Callable, Any

import streamlit as st
import pandas as pd

# Use absolute imports
from llm_evaler.src.utils import sample_for_grading, truncate_text
from llm_evaler.src.config import MIN_GRADES_REQUIRED


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
    if "current_response_id" not in st.session_state:
        # Sample the first response to grade
        sampling_strategy = st.session_state.get("sampling_strategy", "alternating")
        sampled_ids = sample_for_grading(evaluator, num_samples=1, strategy=sampling_strategy)
        
        if sampled_ids:
            st.session_state.current_response_id = sampled_ids[0]
        else:
            st.session_state.current_response_id = None
    
    if "graded_responses" not in st.session_state:
        st.session_state.graded_responses = []
        
    if "criteria_drift_notes" not in st.session_state:
        st.session_state.criteria_drift_notes = {}
        
    if "assertion_feedback" not in st.session_state:
        st.session_state.assertion_feedback = {}
    
    # Update alignment metrics in real-time
    alignment_metrics = evaluator.compute_alignment_metrics() if evaluator.get_grades() else None
    
    # Display real-time alignment insights if we have grades
    if alignment_metrics and len(evaluator.get_grades()) >= 5:
        with st.expander("📊 Real-time Alignment Insights", expanded=True):
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                avg_coverage = sum(m.get("coverage", 0) for m in alignment_metrics.values() if "coverage" in m) / len(alignment_metrics)
                st.metric("Coverage", f"{avg_coverage:.1%}", help="Percentage of human-labeled 'bad' outputs correctly flagged")
            
            with metrics_col2:
                avg_ffr = sum(m.get("ffr", 0) for m in alignment_metrics.values() if "ffr" in m) / len(alignment_metrics) 
                st.metric("False Failure Rate", f"{avg_ffr:.1%}", help="Percentage of human-labeled 'good' outputs incorrectly flagged")
            
            with metrics_col3:
                avg_alignment = sum(m.get("alignment", 0) for m in alignment_metrics.values() if "alignment" in m) / len(alignment_metrics)
                st.metric("Alignment Score", f"{avg_alignment:.1%}", help="Harmonic mean of coverage and (1-FFR)")
            
            # Show top performing variants
            st.markdown("##### Top Performing Assertion Variants")
            
            top_variants = []
            for criterion_name, metrics in alignment_metrics.items():
                if "variant_scores" in metrics:
                    for variant_id, score in metrics["variant_scores"].items():
                        # Extract variant number and description from evaluator assertions
                        variant_info = next((a for a in evaluator.assertions.get(criterion_name, []) if a["id"] == variant_id), {})
                        variant_name = variant_info.get("variant_name", variant_id)
                        variant_description = variant_info.get("description", "")
                        top_variants.append({
                            "Criterion": criterion_name,
                            "Variant": variant_name,
                            "Description": variant_description,
                            "Alignment": score
                        })
            
            if top_variants:
                # Sort by alignment score descending
                top_variants.sort(key=lambda x: x["Alignment"], reverse=True)
                
                # Show top 5
                variant_df = pd.DataFrame(top_variants[:5])
                st.dataframe(
                    variant_df,
                    column_config={
                        "Criterion": st.column_config.TextColumn("Criterion"),
                        "Variant": st.column_config.TextColumn("Variant"),
                        "Description": st.column_config.TextColumn("Approach"),
                        "Alignment": st.column_config.ProgressColumn(
                            "Alignment Score",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                    hide_index=True,
                )
    
    # Allow user to select sampling strategy
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("**Sampling Strategy:**")
    with col2:
        sampling_options = {
            "alternating": "Alternating High/Low Confidence (Recommended)",
            "random": "Random Sampling",
            "low_confidence": "Low Confidence First",
            "high_confidence": "High Confidence First",
            "disagreement": "Highest Assertion Disagreement"
        }
        
        sampling_strategy = st.selectbox(
            "Choose how to select responses for grading:",
            options=list(sampling_options.keys()),
            format_func=lambda x: sampling_options[x],
            index=0,
            label_visibility="collapsed"
        )
        
        # Update session state if changed
        if "sampling_strategy" not in st.session_state or st.session_state.sampling_strategy != sampling_strategy:
            st.session_state.sampling_strategy = sampling_strategy
    
    # Display the current number of grades with information about ongoing refinement
    human_grades = evaluator.get_grades()
    
    # Create a progress indicator for grades
    col1, col2 = st.columns([3, 1])
    with col1:
        # Progress bar
        progress = min(len(human_grades) / MIN_GRADES_REQUIRED, 1.0)
        st.progress(progress, text=f"Grading progress: {len(human_grades)}/{MIN_GRADES_REQUIRED}")
    
    with col2:
        # Current grade count and target
        st.metric(
            "Grades Collected", 
            f"{len(human_grades)}", 
            delta=f"{MIN_GRADES_REQUIRED - len(human_grades)} more needed" if len(human_grades) < MIN_GRADES_REQUIRED else "Sufficient",
            delta_color="inverse"
        )
    
    # "Finish Grading" button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Finish Grading", type="primary", disabled=len(human_grades) < MIN_GRADES_REQUIRED):
            st.session_state.current_response_id = None
            st.success(f"Thanks for grading! You've graded {len(human_grades)} responses. Generating alignment report...")
            # Ensure real-time alignment metrics are updated
            evaluator.compute_alignment_metrics(force_recompute=True)
            st.rerun()
    
    with col2:
        # Option to revisit criteria definition
        if len(st.session_state.criteria_drift_notes) > 0:
            if st.button("Refine Criteria Definition", help="Update your criteria based on observations from grading"):
                # This will take the user back to the criteria definition step
                st.session_state.step = "criteria_definition"
                st.rerun()
    
    # If there's a response to grade
    if st.session_state.current_response_id:
        response = evaluator.get_response_with_results(st.session_state.current_response_id)
        
        if response:
            with st.container(border=True):
                # Display the response for grading
                st.subheader("Question")
                st.write(response["question"])
                
                st.subheader("Options")
                for i, option in enumerate(response["options"]):
                    st.write(f"{chr(65 + i)}) {option}")
                
                # Add a divider to separate question from response
                st.divider()
                
                # Calculate and display confidence score if assertions have been run
                confidence_score = None
                if st.session_state.current_response_id in evaluator.results:
                    result = evaluator.results[st.session_state.current_response_id]
                    passes = 0
                    total = 0
                    
                    for assertion_result in result.get("assertions", {}).values():
                        if assertion_result.get("error") is None:  # Only count assertions without errors
                            total += 1
                            if assertion_result.get("passes", False):
                                passes += 1
                    
                    if total > 0:
                        confidence_score = passes / total
                
                # Show confidence score if available
                confidence_col, sampling_col = st.columns([1, 3])
                
                with confidence_col:
                    if confidence_score is not None:
                        confidence_color = "green" if confidence_score > 0.7 else "orange" if confidence_score > 0.3 else "red"
                        st.markdown(f"""
                        <div style="text-align: center; margin-bottom: 10px;">
                            <span style="color: {confidence_color}; font-weight: bold;">
                                Confidence: {confidence_score:.2f}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                
                with sampling_col:
                    # Explain sampling strategy
                    if st.session_state.sampling_strategy == "alternating":
                        st.info("Selected using alternating high/low confidence sampling", icon="ℹ️")
                    elif st.session_state.sampling_strategy == "low_confidence":
                        st.info("Selected because assertions have low confidence", icon="⚠️")
                    elif st.session_state.sampling_strategy == "high_confidence":
                        st.info("Selected because assertions have high confidence", icon="✅")
                    elif st.session_state.sampling_strategy == "disagreement":
                        st.info("Selected because assertions disagree significantly", icon="🔄")
                    else:
                        st.info("Selected randomly", icon="🎲")
                
                # Display the response
                st.markdown("### LLM Response")
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
                
                # Display assertion results if available
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
                                "Error": truncate_text(result["error"]) if result["error"] else "",
                                "Feedback": st.session_state.assertion_feedback.get(assertion_id, {}).get(st.session_state.current_response_id, "")
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
                                            
                                            # Variant feedback
                                            assertion_id = result["AssertionID"]
                                            if st.button(f"{'👍' if result['Passes'] else '👎'} Accurate", key=f"accurate_{assertion_id}"):
                                                # Record feedback on this assertion variant
                                                if assertion_id not in st.session_state.assertion_feedback:
                                                    st.session_state.assertion_feedback[assertion_id] = {}
                                                
                                                st.session_state.assertion_feedback[assertion_id][st.session_state.current_response_id] = "accurate"
                                                st.rerun()
                                            
                                            if st.button(f"{'👎' if result['Passes'] else '👍'} Inaccurate", key=f"inaccurate_{assertion_id}"):
                                                # Record feedback on this assertion variant
                                                if assertion_id not in st.session_state.assertion_feedback:
                                                    st.session_state.assertion_feedback[assertion_id] = {}
                                                
                                                st.session_state.assertion_feedback[assertion_id][st.session_state.current_response_id] = "inaccurate"
                                                st.rerun()
                                            
                                            # Show current feedback if any
                                            current_feedback = st.session_state.assertion_feedback.get(assertion_id, {}).get(st.session_state.current_response_id, "")
                                            if current_feedback:
                                                st.info(f"Marked as {current_feedback}")
                
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