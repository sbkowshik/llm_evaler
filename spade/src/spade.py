"""SPADE: System for Prompt Analysis and Delta-based Evaluation."""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set

from llm_evaler.src.llm_client import call_openai_api, call_openai_api_async
from llm_evaler.src.utils import ensure_dir
from llm_evaler.src.config import PROMPT_TEMPLATE, DEFAULT_CRITERIA
from llm_evaler.src.evaluator import Evaluator
from llm_evaler.src.alignment import compute_confusion_matrix, compute_alignment_metrics

from spade.src.config import DEFAULT_SPADE_CRITERIA, DEFAULT_FALSE_FAILURE_RATE_THRESHOLD
from spade.src.criteria_generator import get_default_criteria
from spade.src.assertion_generator import generate_assertions, execute_llm_assertion
from spade.src.optimizer import (
    select_minimal_assertion_set, 
    compute_spade_metrics,
    compute_alignment_with_human_grades
)


def load_json_file(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class SPADE:
    """SPADE implementation for automated evaluation of LLM outputs."""
    
    def __init__(
        self,
        prompt_template: str = PROMPT_TEMPLATE,
        ffr_threshold: float = DEFAULT_FALSE_FAILURE_RATE_THRESHOLD,
        data_file: str = "llm_evaler/data/llm_responses.json",
    ):
        """
        Initialize SPADE with a prompt template and parameters.
        
        Args:
            prompt_template: The template used to generate responses
            ffr_threshold: Maximum false failure rate for assertions
            data_file: Path to file containing LLM responses
        """
        self.prompt_template = prompt_template
        self.ffr_threshold = ffr_threshold
        self.data_file = data_file
        
        # Will be populated during execution
        self.responses = []
        self.criteria = []
        self.assertions = []
        self.evaluator = None
        self.evaluator_results = {}
        self.selected_assertions = {}
        self.human_grades = {}
    
    def load_responses(self, file_path: Optional[str] = None) -> List[Dict]:
        """
        Load LLM responses from a file.
        
        Args:
            file_path: Path to the file containing responses
            
        Returns:
            List of response dictionaries
        """
        path = file_path or self.data_file
        self.responses = load_json_file(path)
        return self.responses
    
    def extract_human_grades(self) -> Dict[str, bool]:
        """
        Extract human grades from responses if available.
        Not part of standard SPADE, used for comparison with EvalGen.
        
        Returns:
            Dictionary mapping response_id to human grade
        """
        human_grades = {}
        
        for response in self.responses:
            if "grades" in response and response["grades"]:
                # For simplicity, consider a response good if average grade > 0.5
                grades = [float(g["grade"]) for g in response["grades"] if "grade" in g]
                if grades:
                    avg_grade = sum(grades) / len(grades)
                    human_grades[response["question_id"]] = avg_grade > 0.5
        
        self.human_grades = human_grades
        return human_grades
    
    def generate_criteria(self) -> List[Dict]:
        """
        Use the default criteria from config.py.
        
        Returns:
            List of default criteria dictionaries
        """
        # Use only the default criteria from config.py
        self.criteria = get_default_criteria()
        return self.criteria
    
    def generate_assertions(self) -> List[Dict]:
        """
        Generate assertions for each criterion.
        
        Returns:
            List of assertion dictionaries
        """
        if not self.criteria:
            raise ValueError("No criteria available. Call generate_criteria() first.")
        
        all_assertions = []
        
        # Use a sample of responses for assertion generation
        sample_responses = self.responses[:10]
        
        # Generate assertions for each criterion
        for criterion in self.criteria:
            assertions = generate_assertions(criterion, sample_responses)
            all_assertions.extend(assertions)
        
        self.assertions = all_assertions
        return all_assertions
    
    def initialize_evaluator(self):
        """
        Initialize the evaluator with responses and assertions.
        Must be called after loading responses and generating assertions.
        """
        if not self.responses:
            raise ValueError("No responses available. Call load_responses() first.")
        
        if not self.assertions:
            raise ValueError("No assertions available. Call generate_assertions() first.")
        
        # Format assertions as expected by Evaluator (grouped by criterion_id)
        assertions_by_criterion = {}
        for assertion in self.assertions:
            criterion_id = assertion["criterion_id"]
            if criterion_id not in assertions_by_criterion:
                assertions_by_criterion[criterion_id] = []
            assertions_by_criterion[criterion_id].append(assertion)
        
        # Create the evaluator
        self.evaluator = Evaluator(
            responses=self.responses,
            assertions=assertions_by_criterion
        )
    
    def evaluate_assertions(self, sample_size: Optional[int] = None) -> Dict:
        """
        Evaluate assertions on a sample of responses.
        
        Args:
            sample_size: Number of responses to evaluate (None for all)
            
        Returns:
            Evaluation results dictionary
        """
        if not self.evaluator:
            raise ValueError("Evaluator not initialized. Call initialize_evaluator() first.")
        
        # Run assertions
        print("Running assertions...")
        self.evaluator_results = self.evaluator.run_assertions(show_progress=True)
        
        return self.evaluator_results
    
    def select_assertions(self) -> Dict[str, str]:
        """
        Select optimal set of assertions.
        
        Returns:
            Dictionary mapping criterion_id to assertion_id
        """
        if not self.evaluator_results:
            raise ValueError("No evaluation results available. Call evaluate_assertions() first.")
        
        self.selected_assertions = select_minimal_assertion_set(
            evaluator_results=self.evaluator_results,
            criteria=self.criteria,
            assertions=self.assertions,
            ffr_threshold=self.ffr_threshold,
        )
        
        return self.selected_assertions
    
    def compute_metrics(self) -> Dict:
        """
        Compute metrics for selected assertions.
        
        Returns:
            Dictionary of metrics
        """
        if not self.selected_assertions:
            raise ValueError("No selected assertions available. Call select_assertions() first.")
        
        metrics = compute_spade_metrics(
            evaluator_results=self.evaluator_results,
            selected_assertions=self.selected_assertions,
            responses=self.responses,
        )
        
        # If human grades available, compute alignment metrics
        if self.human_grades:
            alignment_metrics = compute_alignment_with_human_grades(
                evaluator_results=self.evaluator_results,
                selected_assertions=self.selected_assertions,
                human_grades=self.human_grades,
            )
            
            metrics["alignment"] = alignment_metrics
        
        return metrics
    
    def compute_individual_alignment_metrics(self) -> Dict[str, Dict]:
        """
        Compute alignment metrics for each individual assertion.
        
        Returns:
            Dictionary mapping assertion_id to alignment metrics
        """
        if not self.evaluator_results or not self.human_grades:
            return {}
        
        individual_metrics = {}
        
        for assertion in self.assertions:
            assertion_id = assertion["id"]
            
            # Get all results for this assertion
            assertion_results = {}
            for response_id, result in self.evaluator_results.items():
                if (
                    "assertions" in result and 
                    assertion_id in result["assertions"] and 
                    "passes" in result["assertions"][assertion_id]
                ):
                    assertion_results[response_id] = {
                        "passes": result["assertions"][assertion_id]["passes"]
                    }
            
            # Compute confusion matrix and alignment metrics
            confusion_matrix = compute_confusion_matrix(assertion_results, self.human_grades)
            alignment_metrics = compute_alignment_metrics(confusion_matrix)
            
            # Store alignment metrics
            individual_metrics[assertion_id] = alignment_metrics
        
        return individual_metrics
    
    def run_pipeline(
        self,
        sample_size: Optional[int] = None,
        extract_grades: bool = True,
    ) -> Dict:
        """
        Run the full SPADE pipeline.
        
        Args:
            sample_size: Number of responses to evaluate
            extract_grades: Whether to extract human grades
            
        Returns:
            Dictionary of results and metrics
        """
        # Load responses
        print("Loading responses...")
        self.load_responses()
        
        # Extract human grades if available (for comparison with EvalGen)
        if extract_grades:
            print("Extracting human grades...")
            self.extract_human_grades()
        
        # Generate criteria
        print("Using default criteria...")
        self.generate_criteria()
        
        # Generate assertions
        print("Generating assertions...")
        self.generate_assertions()
        
        # Initialize evaluator
        print("Initializing evaluator...")
        self.initialize_evaluator()
        
        # Evaluate assertions
        print("Evaluating assertions...")
        self.evaluate_assertions(sample_size=sample_size)
        
        # Calculate individual assertion alignment metrics
        print("Computing individual assertion alignment metrics...")
        individual_alignment = self.compute_individual_alignment_metrics()
        
        # Select assertions
        print("Selecting optimal assertion set...")
        self.select_assertions()
        
        # Compute metrics
        print("Computing overall metrics...")
        metrics = self.compute_metrics()
        
        return {
            "criteria": self.criteria,
            "assertions": self.assertions,
            "selected_assertions": self.selected_assertions,
            "individual_assertion_alignment": individual_alignment,
            "metrics": metrics,
        }

    def collect_human_grades_streamlit(self, st) -> Dict[str, bool]:
        """
        Collect human grades through a Streamlit interface.
        
        Args:
            st: Streamlit module
            
        Returns:
            Dictionary mapping response_id to human grade
        """
        human_grades = {}
        
        if not self.responses:
            st.error("No responses loaded. Please load responses first.")
            return human_grades
        
        st.subheader("Human Reference Grading")
        st.write("Please grade the following responses:")
        
        # Use session state to preserve grades between reruns
        if "human_grades" not in st.session_state:
            st.session_state.human_grades = {}
        
        # Sample responses to grade if there are too many
        sample_size = min(15, len(self.responses))
        sample_responses = self.responses[:sample_size]
        
        for idx, response in enumerate(sample_responses):
            question_id = response.get("question_id", f"question_{idx}")
            with st.expander(f"Response {idx+1} - ID: {question_id}"):
                st.write("**Question:**")
                st.write(response.get("question", "No question available"))
                
                st.write("**Response:**")
                st.write(response.get("response", "No response available"))
                
                # Create a key for this specific grade
                grade_key = f"grade_{question_id}"
                
                # Initialize from session state or default to None
                if grade_key not in st.session_state:
                    st.session_state[grade_key] = None
                
                # Radio button for grading
                grade = st.radio(
                    "Is this response satisfactory?",
                    options=["Good", "Bad", "Unsure"],
                    key=grade_key,
                    horizontal=True,
                )
                
                # Store grade in session state
                if grade in ["Good", "Bad"]:
                    st.session_state.human_grades[question_id] = (grade == "Good")
        
        # Submit button to finalize grades
        if st.button("Submit Grades"):
            human_grades = st.session_state.human_grades
            self.human_grades = human_grades
            st.success(f"Collected {len(human_grades)} human grades.")
        
        return human_grades


def run_spade_comparison(
    data_file: str = "llm_evaler/data/llm_responses.json",
    sample_size: int = 50,
    ffr_threshold: float = DEFAULT_FALSE_FAILURE_RATE_THRESHOLD,
) -> Dict:
    """
    Run SPADE and return results for comparison with EvalGen.
    
    Args:
        data_file: Path to file containing LLM responses
        sample_size: Number of responses to evaluate
        ffr_threshold: Maximum false failure rate for assertions
        
    Returns:
        Dictionary of results
    """
    # Initialize SPADE
    spade = SPADE(
        prompt_template=PROMPT_TEMPLATE,
        ffr_threshold=ffr_threshold,
        data_file=data_file,
    )
    
    # Run the full pipeline
    results = spade.run_pipeline(
        sample_size=sample_size,
        extract_grades=True,
    )
    
    # Print summary
    print("\nSPADE Evaluation Summary:")
    print(f"Using {len(results['criteria'])} default criteria")
    print(f"Generated {len(results['assertions'])} assertions")
    print(f"Selected {len(results['selected_assertions'])} assertions")
    
    if "alignment" in results["metrics"]:
        alignment = results["metrics"]["alignment"]
        print("\nAlignment Metrics:")
        print(f"Alignment score: {alignment['alignment_score']:.4f}")
        print(f"Coverage: {alignment['coverage']:.4f}")
        print(f"False failure rate (FFR): {alignment['ffr']:.4f}")
        
        # Confusion matrix details
        if "confusion_matrix" in alignment:
            cm = alignment["confusion_matrix"]
            print("\nConfusion Matrix:")
            print(f"True Positive: {cm['true_positive']} (Passes good outputs)")
            print(f"False Positive: {cm['false_positive']} (Passes bad outputs)")
            print(f"True Negative: {cm['true_negative']} (Fails bad outputs)")
            print(f"False Negative: {cm['false_negative']} (Fails good outputs)")
    
    print(f"\nOverall pass rate: {results['metrics']['overall_pass_rate']:.4f}")
    
    # Save results to file
    output_file = "spade/results.json"
    ensure_dir(os.path.dirname(output_file))
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


def create_minimal_streamlit_app():
    """
    Create a minimal Streamlit app for SPADE demonstration.
    This function provides instructions for creating a Streamlit app.
    """
    print("""
To create a minimal Streamlit app for SPADE, create a file named 'spade_app.py' with the following content:

```python
import streamlit as st
import pandas as pd
import json
from spade.src.spade import SPADE, run_spade_comparison

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
```

Run the app with:
```bash
streamlit run spade_app.py
```
""")


if __name__ == "__main__":
    run_spade_comparison() 