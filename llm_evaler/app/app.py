"""Main Streamlit application for LLM evaluation."""

import os
import sys

import streamlit as st

# Import components with relative imports
from llm_evaler.app.components.prompt_node import prompt_node
from llm_evaler.app.components.criteria_wizard import criteria_wizard
from llm_evaler.app.components.grading_interface import grading_interface
from llm_evaler.app.components.report_card import report_card
from llm_evaler.src.evaluator import Evaluator
from llm_evaler.src.config import MIN_GRADES_REQUIRED


def main():
    """Main application function."""
    # Set up the Streamlit page
    st.set_page_config(
        page_title="LLM Evaluator",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("LLM Evaluator")
    st.markdown("""
    This application helps you align automated LLM evaluations with human preferences, following
    the EvalGen methodology to ensure your validators reflect your quality standards.
    
    ### Process:
    1. **Configure the prompt** and generate sample LLM responses
    2. **Define evaluation criteria** (manually or using an LLM)
    3. **Grade responses** to align assertions with your preferences
    4. **Review the report card** showing alignment metrics and selected validators
    """)
    
    # Initialize session state
    if "step" not in st.session_state:
        st.session_state.step = "prompt_config"
    
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = None
    
    # Debug information (collapsible)
    with st.expander("Debug Info", expanded=False):
        # Show session state variables
        st.write("**Session State Variables:**")
        session_state_info = {
            "step": st.session_state.step,
            "evaluator_exists": st.session_state.evaluator is not None,
            "criteria_exists": "criteria" in st.session_state,
            "assertions_exists": "assertions" in st.session_state,
            "criteria_drift_notes": len(st.session_state.get("criteria_drift_notes", {}))
        }
        st.json(session_state_info)
        
        # Show evaluator details if it exists
        if st.session_state.evaluator is not None:
            st.write("**Evaluator Info:**")
            evaluator_info = {
                "num_responses": len(st.session_state.evaluator.responses),
                "has_assertions": hasattr(st.session_state.evaluator, "assertions") and bool(st.session_state.evaluator.assertions),
                "num_grades": len(st.session_state.evaluator.get_grades())
            }
            st.json(evaluator_info)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Display current step
    steps = {
        "prompt_config": "1. Prompt Configuration",
        "criteria_definition": "2. Criteria Definition",
        "response_grading": "3. Response Grading",
        "report_card": "4. Report Card"
    }
    
    current_step = st.session_state.step
    
    # Create a progress indicator
    step_numbers = {step: i+1 for i, step in enumerate(steps.keys())}
    current_step_number = step_numbers[current_step]
    st.sidebar.progress(current_step_number / len(steps))
    
    # Navigation buttons
    for step, label in steps.items():
        button_type = "primary" if step == current_step else "secondary"
        if st.sidebar.button(label, type=button_type, key=f"nav_{step}"):
            # Always allow navigation to the current step or previous steps
            if step == current_step or (
                # Step ordering logic (prompt_config -> criteria_definition -> response_grading -> report_card)
                (step == "prompt_config") or
                (step == "criteria_definition" and current_step in ["response_grading", "report_card"]) or
                (step == "response_grading" and current_step == "report_card")
            ):
                st.session_state.step = step
                st.rerun()
            # Only allow navigation to future steps that are ready
            elif step == "criteria_definition" and st.session_state.evaluator is not None:
                st.session_state.step = step
                st.rerun()
            elif step == "response_grading" and (
                (st.session_state.evaluator is not None and hasattr(st.session_state.evaluator, "assertions") and bool(st.session_state.evaluator.assertions)) or
                ("assertions" in st.session_state and bool(st.session_state.assertions))
            ):
                st.session_state.step = step
                st.rerun()
            elif step == "report_card" and st.session_state.evaluator is not None and bool(st.session_state.evaluator.get_grades()):
                st.session_state.step = step
                st.rerun()
            else:
                st.sidebar.warning(f"Cannot navigate to {label} yet. Complete previous steps first.")
    
    # EvalGen methodology guide in sidebar
    with st.sidebar.expander("About EvalGen Methodology"):
        st.markdown("""
        **EvalGen** is a methodology for aligning automated evaluations with human preferences:

        1. **Criteria Suggestion**: Generate initial evaluation criteria
        2. **Assertion Synthesis**: Create code and LLM-based validators for each criterion
        3. **Human Grading**: Collect feedback on a subset of outputs
        4. **Alignment Calculation**: Select assertions that best match human judgments
        5. **Iterative Refinement**: Support criteria drift and updates
        6. **Transparent Reporting**: Visualize alignment metrics for trust

        This approach ensures validators reflect your standards while reducing manual effort.
        """)
    
    # Callback functions
    def on_responses_generated(responses):
        # Create the evaluator
        st.session_state.evaluator = Evaluator(responses, {})
        
        # Move to the next step
        st.session_state.step = "criteria_definition"
        st.rerun()
    
    def on_criteria_selected(criteria, assertions):
        # Update the evaluator with the assertions
        if st.session_state.evaluator:
            st.session_state.evaluator.assertions = assertions
            
            # Store criteria and assertions in session state
            st.session_state.criteria = criteria
            st.session_state.assertions = assertions
            
            # Run the assertions
            with st.spinner("Running assertions on responses..."):
                st.session_state.evaluator.run_assertions()
            
            # Move to the next step
            st.session_state.step = "response_grading"
            st.rerun()
    
    def on_grade_submitted(response_id, grade):
        # Update alignment after each grade is submitted
        # This callback could be used to re-run assertions or update the UI in real-time
        pass
    
    # Main content based on current step
    if current_step == "prompt_config":
        # Display step description
        st.subheader("Step 1: Configure Prompt and Generate Responses")
        st.write("""
        Configure the prompt and generate sample LLM responses for evaluation.
        These responses will be used for creating evaluation criteria and measuring alignment.
        """)
        
        responses, _ = prompt_node(on_responses_generated=on_responses_generated)
    
    elif current_step == "criteria_definition":
        # Display step description
        st.subheader("Step 2: Define Evaluation Criteria")
        st.write("""
        Define the criteria for evaluating LLM responses. You can:
        - Use an LLM to suggest criteria based on your task
        - Manually define your own criteria
        - Start with grading and refine criteria later
        """)
        
        selected_criteria, assertions = criteria_wizard(
            responses=st.session_state.evaluator.responses,
            on_criteria_selected=on_criteria_selected
        )
    
    elif current_step == "response_grading":
        # Display step description
        st.subheader("Step 3: Grade Responses and Align Assertions")
        st.write("""
        Grade a sample of LLM responses to align the automated validators with your preferences.
        This human feedback acts as ground truth for selecting the best assertions.
        """)
        
        grades = grading_interface(
            evaluator=st.session_state.evaluator,
            on_grade_submitted=on_grade_submitted
        )
        
        # If we have enough grades, enable the report card
        if len(grades) >= MIN_GRADES_REQUIRED:
            st.success(f"You've graded enough responses ({len(grades)}/{MIN_GRADES_REQUIRED}) to generate an alignment report!")
            
            if st.button("View Alignment Report", type="primary"):
                st.session_state.step = "report_card"
                st.rerun()
    
    elif current_step == "report_card":
        # Display step description
        st.subheader("Step 4: Review Alignment Report")
        st.write("""
        This report shows how well the automated validators align with your grading decisions.
        The best assertions for each criterion are selected based on alignment with your preferences.
        """)
        
        selected_assertions, overall_metrics = report_card(
            evaluator=st.session_state.evaluator
        )
        
        # Add options for next steps
        st.subheader("Next Steps")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Option to export the selected validators
            st.write("**Export Selected Validators**")
            st.write("Save the aligned validators for use in your own applications:")
            
            # Prepare Python code for export
            export_code = "# Selected validators from EvalGen alignment\n\n"
            
            for criterion_id, assertion in selected_assertions.items():
                export_code += f"# {criterion_id}\n"
                if assertion["type"] == "code":
                    export_code += assertion["implementation"] + "\n\n"
                else:
                    export_code += f"'''\nLLM assertion prompt template for {criterion_id}:\n"
                    export_code += assertion["implementation"] + "\n'''\n\n"
            
            # Download button for Python code
            st.download_button(
                label="Download as Python Module",
                data=export_code,
                file_name="evalgen_validators.py",
                mime="text/plain"
            )
        
        with col2:
            # Option to revisit criteria
            st.write("**Refine Your Evaluation**")
            st.write("Continue improving your evaluation process:")
            
            # Add more grades button
            if st.button("Add More Human Grades"):
                st.session_state.step = "response_grading"
                st.rerun()
            
            # Update criteria button
            if st.button("Revise Evaluation Criteria"):
                st.session_state.step = "criteria_definition"
                st.rerun()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **LLM Evaluator** implements the EvalGen methodology for aligning LLM evaluations with human preferences.
        
        This helps ensure your automated evaluations reflect your quality standards.
        """
    )


if __name__ == "__main__":
    main() 