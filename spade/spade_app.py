import streamlit as st
import pandas as pd
import json
from spade.src.spade import SPADE

st.set_page_config(page_title="SPADE Evaluation", layout="wide")

st.title("SPADE: System for Prompt Analysis and Delta-based Evaluation")

# Initialize SPADE in session state if not already done
if 'spade' not in st.session_state:
    st.session_state.spade = SPADE()

spade = st.session_state.spade

# Sidebar for configuration
st.sidebar.header("Configuration")
data_file = st.sidebar.text_input("Data file path", value="llm_evaler/data/llm_responses.json")
sample_size = st.sidebar.slider("Sample size", min_value=10, max_value=100, value=50, step=10)
ffr_threshold = st.sidebar.slider("False Failure Rate Threshold", min_value=0.1, max_value=0.9, value=0.4, step=0.1)

# Multi-step process with tabs
tabs = st.tabs(["Load Data", "Human Grading", "Run Evaluation", "Results"])

with tabs[0]:
    st.header("Load LLM Responses")
    if st.button("Load Responses"):
        with st.spinner("Loading responses..."):
            spade.data_file = data_file
            responses = spade.load_responses()
            st.success(f"Loaded {len(responses)} responses.")
            st.session_state.data_loaded = True

with tabs[1]:
    st.header("Human Reference Grading")
    if not st.session_state.get('data_loaded', False):
        st.warning("Please load data first in the 'Load Data' tab.")
    else:
        human_grades = spade.collect_human_grades_streamlit(st)
        if human_grades:
            st.session_state.human_grades_collected = True

with tabs[2]:
    st.header("Run SPADE Evaluation")
    if not st.session_state.get('data_loaded', False):
        st.warning("Please load data first in the 'Load Data' tab.")
    else:
        use_human_grades = st.checkbox("Use human grades for comparison", 
                                       value=st.session_state.get('human_grades_collected', False))
        
        if st.button("Run Evaluation"):
            with st.spinner("Running SPADE evaluation..."):
                # Run individual steps of the pipeline
                spade.generate_criteria()
                spade.generate_assertions()
                spade.initialize_evaluator()
                spade.evaluate_assertions(sample_size=sample_size)
                spade.select_assertions()
                
                # Compute metrics
                if use_human_grades and st.session_state.get('human_grades_collected', False):
                    individual_alignment = spade.compute_individual_alignment_metrics()
                
                metrics = spade.compute_metrics()
                
                # Store results
                st.session_state.results = {
                    "criteria": spade.criteria,
                    "assertions": spade.assertions,
                    "selected_assertions": spade.selected_assertions,
                    "metrics": metrics,
                }
                
                if use_human_grades and st.session_state.get('human_grades_collected', False):
                    st.session_state.results["individual_assertion_alignment"] = individual_alignment
                
                st.success("Evaluation completed successfully!")
                st.session_state.evaluation_done = True

with tabs[3]:
    st.header("Evaluation Results")
    if not st.session_state.get('evaluation_done', False):
        st.info("Run the evaluation first to see results here.")
    else:
        results = st.session_state.results
        
        # Display metrics
        st.subheader("Evaluation Metrics")
        metrics_container = st.container()
        
        col1, col2, col3 = metrics_container.columns(3)
        
        if "alignment" in results["metrics"]:
            alignment = results["metrics"]["alignment"]
            col1.metric("Alignment Score", f"{alignment['alignment_score']:.4f}")
            col2.metric("Coverage", f"{alignment['coverage']:.4f}")
            col3.metric("False Failure Rate", f"{alignment['ffr']:.4f}")
        
        st.metric("Overall Pass Rate", f"{results['metrics']['overall_pass_rate']:.4f}")
        
        # Display confusion matrix if available
        if "alignment" in results["metrics"] and "confusion_matrix" in results["metrics"]["alignment"]:
            st.subheader("Confusion Matrix")
            cm = results["metrics"]["alignment"]["confusion_matrix"]
            
            # Format confusion matrix for display
            cm_df = pd.DataFrame({
                "": ["Human: Good", "Human: Bad"],
                "Assertion: Pass": [cm["true_positive"], cm["false_positive"]],
                "Assertion: Fail": [cm["false_negative"], cm["true_negative"]]
            }).set_index("")
            
            st.table(cm_df)
        
        # Display selected assertions
        st.subheader("Selected Assertions")
        for criterion_id, assertion_id in results["selected_assertions"].items():
            st.write(f"**Criterion: {criterion_id}**")
            # Find the assertion details
            assertion = next((a for a in results["assertions"] if a["id"] == assertion_id), None)
            if assertion:
                st.code(assertion["implementation"], language="python")
        
        # Option to save results
        if st.button("Save Results"):
            output_file = "spade/results.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            st.success(f"Results saved to {output_file}")

# Show instructions for first-time users
if not st.session_state.get('data_loaded', False):
    st.info("Start by loading the LLM responses in the 'Load Data' tab.") 