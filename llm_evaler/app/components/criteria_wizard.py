"""Criteria wizard component for selecting and configuring evaluation criteria."""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Callable

import streamlit as st

# Use absolute imports
from llm_evaler.src.criteria_generator import (
    get_default_criteria,
    generate_criteria_with_llm,
)
from llm_evaler.src.assertion_generator import generate_assertions


def infer_criteria(task_description: str, example_responses: List[Dict], num_criteria: int = 5, include_defaults: bool = True) -> List[Dict]:
    """
    Infer evaluation criteria using an LLM.
    
    Args:
        task_description: Description of the task
        example_responses: Example responses to analyze
        num_criteria: Number of criteria to generate
        include_defaults: Whether to include default criteria
        
    Returns:
        List of inferred criteria
    """
    with st.spinner("Generating criteria suggestions..."):
        # Generate criteria using the LLM
        criteria = generate_criteria_with_llm(
            task_description=task_description,
            examples=example_responses,
            num_criteria=num_criteria,
            include_defaults=False  # Handle defaults separately
        )
        
        # Limit to exactly the requested number
        criteria = criteria[:num_criteria]
        
        # Combine with default criteria if requested
        if include_defaults:
            default_criteria = get_default_criteria()
            # Check for duplicates
            existing_names = [c["name"].lower() for c in criteria]
            for default in default_criteria:
                if default["name"].lower() not in existing_names:
                    criteria.append(default)
        
        return criteria


def criteria_wizard(
    responses: List[Dict],
    on_criteria_selected: Optional[Callable] = None
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """
    Criteria wizard component for selecting and configuring evaluation criteria.
    
    This component helps users define, refine, and update evaluation criteria,
    addressing criteria drift by supporting iterative updates based on human feedback.
    
    Args:
        responses: List of LLM responses to evaluate
        on_criteria_selected: Callback function to call when criteria are selected
        
    Returns:
        Tuple of (selected_criteria, generated_assertions)
    """
    st.header("Evaluation Criteria")
    
    st.markdown("""
    Define the criteria for evaluating LLM responses. You can generate suggestions using an LLM,
    start with a manual configuration, or refine criteria after grading some responses.
    """)
    
    # Check if we already have criteria and assertions in session state
    if "criteria" not in st.session_state or "assertions" not in st.session_state:
        st.session_state.criteria = []
        st.session_state.assertions = {}
        st.session_state.wizard_step = "start"
    
    # Add criteria_history if it doesn't exist (for tracking criteria evolution)
    if "criteria_history" not in st.session_state:
        st.session_state.criteria_history = []
    
    # Check for criteria drift notes from the grading interface
    has_criteria_drift = (
        "criteria_drift_notes" in st.session_state and 
        st.session_state.criteria_drift_notes
    )
    
    # Display criteria drift notification if coming from grading
    if has_criteria_drift and st.session_state.wizard_step not in ["start", "criteria_review_with_drift"]:
        criteria_count = len(st.session_state.criteria_drift_notes)
        st.info(f"💡 You identified {criteria_count} potential criteria improvements during grading. You can incorporate these into your evaluation framework.")
        
        # Show criteria drift notes in an expander
        with st.expander("View criteria observations from grading", expanded=True):
            for response_id, note in st.session_state.criteria_drift_notes.items():
                # Get a short sample of the response for context
                response_text = next((r["response"][:100] + "..." for r in responses if r["id"] == response_id), "")
                
                st.markdown(f"**Note:** {note}")
                st.markdown(f"**From response:** {response_text}")
                st.divider()
            
            if st.button("Refine criteria with these observations", type="primary"):
                # Save current criteria to history before modifications
                if st.session_state.criteria:
                    st.session_state.criteria_history.append({
                        "version": len(st.session_state.criteria_history) + 1,
                        "criteria": st.session_state.criteria.copy(),
                        "timestamp": "Now",
                        "source": "Before criteria drift refinement"
                    })
                
                st.session_state.wizard_step = "criteria_review_with_drift"
                st.rerun()
    
    # Wizard steps
    if st.session_state.wizard_step == "start":
        # Show the initial options
        st.subheader("Choose an option to get started")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Infer Criteria", help="Use an LLM to suggest criteria based on your task"):
                st.session_state.wizard_step = "infer_criteria"
                st.rerun()
        
        with col2:
            if st.button("Manual Configuration", help="Start with some default criteria that you can edit"):
                st.session_state.wizard_step = "manual_config"
                st.session_state.criteria = get_default_criteria()
                st.rerun()
        
        with col3:
            if st.button("Grade First", help="Start by grading some responses to get a better sense of quality issues"):
                st.session_state.wizard_step = "grade_first"
                st.rerun()
    
    elif st.session_state.wizard_step == "infer_criteria":
        st.subheader("Let's infer criteria for your task")
        
        # Task description with detailed instructions
        st.markdown("""
        Provide a detailed description of your task to help the LLM suggest relevant evaluation criteria.
        Include:
        - The purpose of the LLM responses
        - Expected format or structure
        - Important aspects of quality
        - Any specific requirements
        """)
        
        # Task description
        task_description = st.text_area(
            "Task Description",
            value="Evaluate LLM responses to multiple-choice questions from the MMLU dataset. The LLM should select the correct option and provide a clear explanation.",
            height=150
        )
        
        # Generation options
        st.subheader("Generation Options")
        # Replace slider with text input for number of criteria
        num_criteria_str = st.text_input("Number of criteria to generate", value="5")
        try:
            num_criteria = int(num_criteria_str)
            if num_criteria < 1:
                st.warning("Please enter a positive number for criteria count.")
                num_criteria = 5
        except ValueError:
            st.warning("Please enter a valid number for criteria count.")
            num_criteria = 5
            
        include_defaults = st.checkbox("Include default criteria", value=True)
        
        # Generate criteria button
        if st.button("Generate Criteria", type="primary"):
            with st.spinner("Analyzing task and generating criteria suggestions..."):
                # Infer criteria
                generated_criteria = generate_criteria_with_llm(
                    task_description=task_description,
                    examples=responses[:3],
                    num_criteria=num_criteria,
                    include_defaults=False  # Always set to False here, we'll handle defaults separately
                )
                
                # Limit to exactly the requested number
                generated_criteria = generated_criteria[:num_criteria]
                
                # Set criteria in the session state
                if include_defaults:
                    defaults = get_default_criteria()
                    # Check for duplicates before adding defaults
                    existing_names = [c["name"].lower() for c in generated_criteria]
                    default_criteria = []
                    for default in defaults:
                        if default["name"].lower() not in existing_names:
                            default_criteria.append(default)
                    
                    st.session_state.criteria = generated_criteria + default_criteria
                else:
                    st.session_state.criteria = generated_criteria
                
                # Save to criteria history
                st.session_state.criteria_history.append({
                    "version": len(st.session_state.criteria_history) + 1,
                    "criteria": st.session_state.criteria.copy(),
                    "timestamp": "Now",
                    "source": "Initial LLM generation"
                })
                
                # Move to criteria review
                st.session_state.wizard_step = "criteria_review"
                st.rerun()
        
        # Back button
        if st.button("Back"):
            st.session_state.wizard_step = "start"
            st.rerun()
    
    elif st.session_state.wizard_step == "manual_config":
        st.subheader("Manual Criteria Configuration")
        
        st.markdown("""
        Add, edit, or remove criteria to define what makes a good LLM response.
        Each criterion should describe a specific aspect of quality that can be evaluated.
        """)
        
        # Show default criteria with option to edit
        st.session_state.criteria = manual_criteria_editor(st.session_state.criteria)
        
        # Continue button
        if st.button("Continue to Assertion Generation", type="primary"):
            # Save to criteria history
            st.session_state.criteria_history.append({
                "version": len(st.session_state.criteria_history) + 1,
                "criteria": st.session_state.criteria.copy(),
                "timestamp": "Now",
                "source": "Manual configuration"
            })
            
            st.session_state.wizard_step = "generate_assertions"
            st.rerun()
        
        # Back button
        if st.button("Back"):
            st.session_state.wizard_step = "start"
            st.rerun()
    
    elif st.session_state.wizard_step == "grade_first":
        st.info("""
        Starting with grading can help you develop a better understanding of response quality
        before defining formal criteria.
        
        After grading a few responses, you can return here to define criteria based on your observations.
        """)
        
        # Generate minimum criteria for grading
        if st.button("Generate Basic Criteria and Start Grading", type="primary"):
            # Create minimal criteria set
            st.session_state.criteria = [
                {
                    "name": "Overall Quality",
                    "description": "Does this response meet your overall quality expectations?",
                    "implementation_type": "llm"
                }
            ]
            
            # Save to criteria history
            st.session_state.criteria_history.append({
                "version": len(st.session_state.criteria_history) + 1,
                "criteria": st.session_state.criteria.copy(),
                "timestamp": "Now",
                "source": "Initial minimal set for grading"
            })
            
            # Generate basic assertions
            with st.spinner("Generating basic assertions..."):
                assertions = {}
                for criterion in st.session_state.criteria:
                    assertions[criterion["name"]] = generate_assertions(
                        criterion=criterion,
                        example_responses=responses[:3],
                        use_llm=True
                    )
                
                st.session_state.assertions = assertions
                
                if on_criteria_selected:
                    on_criteria_selected(st.session_state.criteria, st.session_state.assertions)
            
            # Store criteria and assertions in session state for later reference
            st.session_state.initial_grade_first_criteria = st.session_state.criteria.copy()
            st.session_state.initial_grade_first_assertions = st.session_state.assertions.copy()
    
    elif st.session_state.wizard_step == "criteria_review_with_drift":
        st.subheader("Refine Criteria Based on Grading Observations")
        
        st.markdown("""
        Update your criteria based on observations from grading. You can:
        - Add new criteria identified during grading
        - Modify existing criteria to better reflect your quality standards
        - Deactivate criteria that aren't relevant
        """)
        
        # Show notes to help with refinement
        with st.expander("Grading observations", expanded=True):
            for response_id, note in st.session_state.criteria_drift_notes.items():
                st.markdown(f"- {note}")
        
        # Use LLM to help refine criteria
        if st.button("Suggest refined criteria using LLM"):
            with st.spinner("Analyzing observations and refining criteria..."):
                # Combine existing criteria and notes into a prompt
                criteria_text = "\n".join([f"- {c['name']}: {c['description']}" for c in st.session_state.criteria])
                notes_text = "\n".join([f"- {note}" for note in st.session_state.criteria_drift_notes.values()])
                
                task_description = f"""
                I need to refine my criteria for evaluating LLM responses to multiple-choice questions.
                
                Current criteria:
                {criteria_text}
                
                Observations from grading:
                {notes_text}
                
                Please suggest a revised set of criteria that addresses these observations.
                """
                
                # Generate refined criteria
                refined_criteria = generate_criteria_with_llm(
                    task_description=task_description,
                    examples=responses[:3],
                    num_criteria=len(st.session_state.criteria) + 2,  # Add a couple more
                    include_defaults=False
                )
                
                # Update criteria
                st.session_state.criteria = refined_criteria
                
                # Save to criteria history
                st.session_state.criteria_history.append({
                    "version": len(st.session_state.criteria_history) + 1,
                    "criteria": st.session_state.criteria.copy(),
                    "timestamp": "Now",
                    "source": "LLM refinement from grading observations"
                })
                
                st.rerun()
        
        # Manual edit option
        st.session_state.criteria = manual_criteria_editor(st.session_state.criteria)
        
        # Continue button
        if st.button("Continue with Refined Criteria", type="primary"):
            # Clear criteria drift notes now that they've been incorporated
            st.session_state.criteria_drift_notes = {}
            
            # Save to criteria history if not already saved
            last_history = st.session_state.criteria_history[-1] if st.session_state.criteria_history else None
            if not last_history or last_history["criteria"] != st.session_state.criteria:
                st.session_state.criteria_history.append({
                    "version": len(st.session_state.criteria_history) + 1,
                    "criteria": st.session_state.criteria.copy(),
                    "timestamp": "Now",
                    "source": "Manual refinement from grading observations"
                })
            
            # Move to assertion generation
            st.session_state.wizard_step = "generate_assertions"
            st.rerun()
    
    elif st.session_state.wizard_step == "criteria_review":
        st.subheader("Review Generated Criteria")
        
        st.markdown("""
        Review and edit the generated criteria before creating assertions.
        You can add, modify, or remove criteria to better match your evaluation needs.
        """)
        
        # Edit criteria
        st.session_state.criteria = manual_criteria_editor(st.session_state.criteria)
        
        # Continue button
        if st.button("Continue to Assertion Generation", type="primary"):
            st.session_state.wizard_step = "generate_assertions"
            st.rerun()
        
        # Back button
        if st.button("Back to Task Description"):
            st.session_state.wizard_step = "infer_criteria"
            st.rerun()
    
    elif st.session_state.wizard_step == "generate_assertions":
        st.subheader("Generate Assertion Variants")
        
        st.markdown("""
        For each criterion, we'll generate multiple LLM-based assertion variants.
        These variants use different prompting strategies to evaluate the same criterion,
        helping identify the approach that best aligns with your judgments.
        """)
        
        # Show criteria being evaluated
        with st.expander("Your evaluation criteria", expanded=True):
            for i, criterion in enumerate(st.session_state.criteria):
                st.markdown(f"**{i+1}. {criterion['name']}**")
                st.markdown(f"{criterion['description']}")
            
        # Options for assertion generation
        num_variants = st.slider("Number of variants per criterion", 3, 5, 5)
        
        # Generate assertions button
        if st.button("Generate Assertion Variants", type="primary"):
            with st.spinner("Generating assertion variants for each criterion..."):
                assertions = {}
                
                # Generate assertions for each criterion
                for criterion in st.session_state.criteria:
                    criterion_assertions = generate_assertions(
                        criterion=criterion,
                        example_responses=responses[:3],
                        use_llm=True
                    )
                    
                    # Limit to requested number of variants
                    assertions[criterion["name"]] = criterion_assertions[:num_variants]
                
                # Store assertions in session state
                st.session_state.assertions = assertions
                
                # Display generated assertions
                st.session_state.wizard_step = "assertion_review"
                st.rerun()
        
        # Back button
        if st.button("Back to Criteria Definition"):
            st.session_state.wizard_step = "criteria_review"
            st.rerun()
    
    elif st.session_state.wizard_step == "assertion_review":
        st.subheader("Review Generated Assertion Variants")
        
        st.markdown("""
        Review the generated assertion variants for each criterion. 
        These will be evaluated against your human grades to select the best-aligned variants.
        """)
        
        # For each criterion, show the assertion variants
        for criterion_name, criterion_assertions in st.session_state.assertions.items():
            st.markdown(f"### {criterion_name}")
            
            # Create tabs for each variant
            tabs = st.tabs([f"Variant {i+1}" for i in range(len(criterion_assertions))])
            
            for i, (tab, assertion) in enumerate(zip(tabs, criterion_assertions)):
                with tab:
                    st.markdown(f"**{assertion.get('variant_name', f'Variant {i+1}')}**")
                    st.markdown(f"*{assertion.get('description', 'No description')}*")
                    
                    with st.expander("View prompt template"):
                        st.code(assertion['implementation'], language="text")
        
        # Continue button
        if st.button("Continue to Grading", type="primary"):
            if on_criteria_selected:
                on_criteria_selected(st.session_state.criteria, st.session_state.assertions)
        
        # Back button
        if st.button("Back to Assertion Generation"):
            st.session_state.wizard_step = "generate_assertions"
            st.rerun()
    
    # Criteria history explorer
    if st.session_state.criteria_history:
        with st.expander("View criteria evolution history", expanded=False):
            st.markdown("### Criteria Evolution")
            st.markdown("Track how your criteria have evolved throughout the evaluation process.")
            
            for version in st.session_state.criteria_history:
                st.markdown(f"**Version {version['version']}** - {version['source']}")
                
                # Replace nested expander with a container + divider
                with st.container():
                    st.markdown(f"**Criteria in version {version['version']}:**")
                    for i, criterion in enumerate(version['criteria']):
                        st.markdown(f"**{i+1}. {criterion['name']}**")
                        st.markdown(f"{criterion['description']}")
                    st.divider()
    
    # Return the current criteria and assertions
    return st.session_state.criteria, st.session_state.assertions


def manual_criteria_editor(criteria: List[Dict]) -> List[Dict]:
    """
    Component for manually editing criteria.
    
    Args:
        criteria: List of criteria to edit
        
    Returns:
        Updated list of criteria
    """
    # Add a new criterion button
    if st.button("+ Add New Criterion"):
        criteria.append({
            "name": "New Criterion",
            "description": "Describe what makes a response pass or fail this criterion",
            "implementation_type": "llm",
            "active": True
        })
    
    # Edit each criterion
    updated_criteria = []
    for i, criterion in enumerate(criteria):
        with st.expander(f"{criterion['name']}", expanded=i == len(criteria) - 1):
            # Criterion name and description
            name = st.text_input("Name", value=criterion["name"], key=f"name_{i}")
            description = st.text_area("Description", value=criterion["description"], key=f"desc_{i}", height=100)
            
            # Status (active/inactive)
            active = st.checkbox("Active", value=criterion.get("active", True), key=f"active_{i}",
                                help="Deactivate instead of deleting to preserve history")
            
            # Delete button
            col1, col2 = st.columns([5, 1])
            with col2:
                delete = st.button("🗑️ Delete", key=f"delete_{i}")
                
            if not delete:
                updated_criteria.append({
                    "name": name,
                    "description": description,
                    "implementation_type": "llm",  # Always LLM-based
                    "active": active
                })
    
    return updated_criteria 