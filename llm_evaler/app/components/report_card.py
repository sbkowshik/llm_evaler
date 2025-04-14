"""Report card component for displaying alignment metrics and visualizations."""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Use absolute imports
from llm_evaler.src.alignment import (
    select_best_assertions,
    compute_overall_alignment,
)
from llm_evaler.src.utils import format_percentage
from llm_evaler.src.config import DEFAULT_FALSE_FAILURE_RATE_THRESHOLD


def plot_confusion_matrix(confusion_matrix: Dict, title: str = "") -> plt.Figure:
    """
    Plot a confusion matrix as a heatmap.
    
    Args:
        confusion_matrix: Dictionary with confusion matrix values
        title: Title for the plot
        
    Returns:
        Matplotlib figure
    """
    # Handle None or non-dict confusion matrix
    if confusion_matrix is None or not isinstance(confusion_matrix, dict):
        confusion_matrix = {}
    
    # Set default values of 0 for any missing keys
    tp = confusion_matrix.get("true_positive", 0)
    fn = confusion_matrix.get("false_negative", 0)
    fp = confusion_matrix.get("false_positive", 0)
    tn = confusion_matrix.get("true_negative", 0)
    
    # Create the confusion matrix as a 2x2 array
    cm = [
        [tp, fn],
        [fp, tn]
    ]
    
    # Create a Pandas DataFrame for better labeling
    cm_df = pd.DataFrame(
        cm,
        index=["Passes", "Fails"],
        columns=["Human: Good", "Human: Bad"]
    )
    
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Plot the heatmap
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax
    )
    
    # Set the title
    if title:
        ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_metrics_comparison(metrics_data: List[Dict]) -> plt.Figure:
    """
    Plot a comparison of metrics across assertions.
    
    Args:
        metrics_data: List of dictionaries with assertion metrics
        
    Returns:
        Matplotlib figure
    """
    # Check if we have any data
    if not metrics_data:
        # Create a dummy figure if no data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No metrics data available", ha='center', va='center')
        return fig
    
    # Extract data for plotting with safe access
    criterion_names = [m.get("name", f"Criterion {i}") for i, m in enumerate(metrics_data)]
    
    # Safely extract metrics with default values
    coverage = []
    ffr = []
    alignment = []
    
    for m in metrics_data:
        metrics_dict = m.get("metrics", {})
        if not isinstance(metrics_dict, dict):
            metrics_dict = {}
        
        coverage.append(metrics_dict.get("coverage", 0))
        ffr.append(metrics_dict.get("false_failure_rate", 0))
        alignment.append(metrics_dict.get("alignment_score", 0))
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set the positions and width for the bars
    pos = np.arange(len(criterion_names))
    width = 0.25
    
    # Create bars
    plt.bar(pos - width, coverage, width, alpha=0.8, color='#66b3ff', label='Coverage')
    plt.bar(pos, 1 - np.array(ffr), width, alpha=0.8, color='#99ff99', label='1 - FFR')
    plt.bar(pos + width, alignment, width, alpha=0.8, color='#ff9966', label='Alignment')
    
    # Add labels, title, and custom x-axis
    plt.ylabel('Score')
    plt.title('Metrics by Criterion')
    plt.xticks(pos, criterion_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    # Add legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add horizontal grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_variant_comparison(evaluator, metrics: Dict[str, Dict]) -> plt.Figure:
    """
    Create a bar chart comparing different assertion variants for each criterion.
    
    Args:
        evaluator: Evaluator instance
        metrics: Dictionary of metrics for each criterion
        
    Returns:
        Matplotlib figure
    """
    # Handle None metrics
    if metrics is None:
        metrics = {}
        
    # Collect data for all criteria
    data = []
    
    for criterion_name, criterion_metrics in metrics.items():
        if "variant_scores" in criterion_metrics:
            for variant_id, score in criterion_metrics["variant_scores"].items():
                # Get variant info
                variant_info = next((a for a in evaluator.assertions.get(criterion_name, []) if a["id"] == variant_id), {})
                variant_name = variant_info.get("variant_name", "Unknown")
                variant_description = variant_info.get("description", "Unknown")
                
                data.append({
                    "Criterion": criterion_name,
                    "Variant": variant_name,
                    "Description": variant_description,
                    "Alignment": score,
                    "Selected": variant_id == criterion_metrics.get("best_assertion_id", "")
                })
    
    if not data:
        # Create a dummy figure if no data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No variant data available", ha='center', va='center')
        return fig
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create a figure with subplots for each criterion
    criteria = df["Criterion"].unique()
    fig, axes = plt.subplots(len(criteria), 1, figsize=(12, 4 * len(criteria)))
    
    # If there's only one criterion, axes won't be an array
    if len(criteria) == 1:
        axes = [axes]
    
    # Plot each criterion in its own subplot
    for i, criterion in enumerate(criteria):
        ax = axes[i]
        criterion_df = df[df["Criterion"] == criterion].sort_values(by="Alignment", ascending=False)
        
        # Create bars
        bars = ax.bar(
            criterion_df["Variant"],
            criterion_df["Alignment"],
            color=[("#ff9966" if selected else "#66b3ff") for selected in criterion_df["Selected"]]
        )
        
        # Add labels and title
        ax.set_title(f"{criterion} - Variant Comparison")
        ax.set_ylabel("Alignment Score")
        ax.set_ylim(0, 1.0)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.02,
                f"{height:.2f}",
                ha='center',
                va='bottom'
            )
        
        # Highlight selected variant
        selected_variants = criterion_df[criterion_df["Selected"]]
        if not selected_variants.empty:
            for _, row in selected_variants.iterrows():
                ax.text(
                    row.name,
                    row["Alignment"] / 2,
                    "Selected",
                    ha='center',
                    va='center',
                    color='white',
                    fontweight='bold'
                )
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def create_assertion_heatmap(evaluator, selected_assertions: Dict[str, Dict]) -> plt.Figure:
    """
    Create a heatmap showing assertion results vs human grades.
    
    Args:
        evaluator: Evaluator instance
        selected_assertions: Dictionary of selected assertions
        
    Returns:
        Matplotlib figure
    """
    # Get human grades
    human_grades = evaluator.get_grades()
    
    # Prepare data for the heatmap
    data = []
    
    # Get response IDs that have human grades
    response_ids = list(human_grades.keys())
    
    # For each response, collect assertion results
    for response_id in response_ids:
        row = {'response_id': response_id, 'human_grade': 'Good' if human_grades[response_id] else 'Bad'}
        
        # Add assertion results
        for criterion_id, assertion in selected_assertions.items():
            assertion_id = assertion["id"]
            
            # Get the result if available
            if response_id in evaluator.results and assertion_id in evaluator.results[response_id]["assertions"]:
                result = evaluator.results[response_id]["assertions"][assertion_id]
                row[criterion_id] = 'Pass' if result["passes"] else 'Fail'
            else:
                row[criterion_id] = 'N/A'
        
        data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Prepare for visualization
    # First, sort by human grade (bad at the top)
    df = df.sort_values(by='human_grade', ascending=False)
    
    # Create a heatmap matrix
    heatmap_data = []
    criterion_ids = list(selected_assertions.keys())
    
    for _, row in df.iterrows():
        human_grade = row['human_grade']
        heatmap_row = []
        
        for criterion_id in criterion_ids:
            if row[criterion_id] == 'Pass':
                # Green if passing
                heatmap_row.append(1)
            elif row[criterion_id] == 'Fail':
                # Red if failing
                heatmap_row.append(0)
            else:
                # Gray for N/A
                heatmap_row.append(0.5)
        
        heatmap_data.append(heatmap_row)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.3)))
    
    # Create custom colormap: red for fail, green for pass, gray for N/A
    cmap = plt.cm.get_cmap('RdYlGn', 3)
    
    # Create the heatmap
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        annot=False,
        cbar=False,
        ax=ax,
        yticklabels=['Bad' if grade == 'Bad' else 'Good' for grade in df['human_grade']],
        xticklabels=criterion_ids
    )
    
    # Add a title
    ax.set_title('Assertion Results vs Human Grades')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def create_criteria_evolution_summary(evaluator) -> Dict[str, Any]:
    """
    Create a summary of how criteria evolved during the evaluation process.
    
    Args:
        evaluator: Evaluator instance
        
    Returns:
        Dictionary with criteria evolution data
    """
    if "criteria_history" not in st.session_state:
        return {
            "versions": 0,
            "criteria_count_initial": 0,
            "criteria_count_final": 0,
            "changes": []
        }
    
    history = st.session_state.criteria_history
    
    if not history:
        return {
            "versions": 0,
            "criteria_count_initial": 0,
            "criteria_count_final": 0,
            "changes": []
        }
    
    # Extract key statistics
    initial_version = history[0]
    final_version = history[-1]
    
    # Calculate changes between versions
    changes = []
    for i in range(1, len(history)):
        prev_version = history[i-1]
        curr_version = history[i]
        
        # Get criteria names for comparison
        prev_names = {c["name"] for c in prev_version["criteria"]}
        curr_names = {c["name"] for c in curr_version["criteria"]}
        
        # Find additions and removals
        added = curr_names - prev_names
        removed = prev_names - curr_names
        
        # Track modifications by comparing descriptions
        prev_criteria = {c["name"]: c for c in prev_version["criteria"]}
        curr_criteria = {c["name"]: c for c in curr_version["criteria"]}
        
        modified = set()
        for name in prev_names & curr_names:
            if prev_criteria[name]["description"] != curr_criteria[name]["description"]:
                modified.add(name)
        
        changes.append({
            "from_version": i,
            "to_version": i + 1,
            "source": curr_version["source"],
            "added_count": len(added),
            "added": list(added),
            "removed_count": len(removed),
            "removed": list(removed),
            "modified_count": len(modified),
            "modified": list(modified)
        })
    
    return {
        "versions": len(history),
        "criteria_count_initial": len(initial_version["criteria"]),
        "criteria_count_final": len(final_version["criteria"]),
        "changes": changes
    }


def create_selected_assertions_summary(evaluator, selected_assertions: Dict[str, Dict], metrics: Dict[str, Dict]) -> List[Dict]:
    """
    Create a detailed summary of the selected assertions and their metrics.
    
    Args:
        evaluator: Evaluator instance
        selected_assertions: Dictionary of selected assertions
        metrics: Dictionary of metrics for each criterion
        
    Returns:
        List of dictionaries with assertion details
    """
    summary = []
    
    for criterion_name, assertion in selected_assertions.items():
        criterion_metrics = metrics.get(criterion_name, {})
        
        # Find the actual assertion object to get more details
        assertion_info = next(
            (a for a in evaluator.assertions.get(criterion_name, []) if a["id"] == assertion["id"]),
            {}
        )
        
        summary.append({
            "criterion": criterion_name,
            "assertion_id": assertion["id"],
            "variant_name": assertion_info.get("variant_name", "Unknown"),
            "description": assertion_info.get("description", "Unknown"),
            "coverage": criterion_metrics.get("coverage", 0),
            "false_failure_rate": criterion_metrics.get("false_failure_rate", 0),
            "alignment_score": criterion_metrics.get("alignment_score", 0),
            "implementation": assertion_info.get("implementation", ""),
            "total_variants": len(evaluator.assertions.get(criterion_name, []))
        })
    
    return summary


def report_card(
    evaluator,
    ffr_threshold: float = DEFAULT_FALSE_FAILURE_RATE_THRESHOLD,
    download_enabled: bool = True
) -> Tuple[Dict[str, Dict], Dict]:
    """
    Report card component for displaying alignment metrics and visualizations.
    
    Args:
        evaluator: Evaluator instance
        ffr_threshold: Maximum False Failure Rate (FFR) threshold
        download_enabled: Whether to enable download of selected assertions
        
    Returns:
        Tuple of (selected_assertions, metrics)
    """
    st.header("EvalGen Report Card")
    
    st.markdown("""
    This report card shows how well different assertion implementations align with your human judgments.
    It provides transparency into the evaluation process and helps you understand the strengths and
    limitations of the selected assertions.
    """)
    
    # Check if we have enough data
    human_grades = evaluator.get_grades()
    
    if not human_grades:
        st.warning("No human grades found. Please grade some responses first.")
        return {}, {}
    
    # Compute alignment metrics and select the best assertions
    try:
        metrics = evaluator.compute_alignment_metrics(force_recompute=True)
    except Exception as e:
        st.error(f"Error computing alignment metrics: {str(e)}")
        metrics = {}
    
    # Select best assertions
    try:
        selected_assertions = select_best_assertions(
            evaluator,
            ffr_threshold=ffr_threshold
        )
    except Exception as e:
        st.error(f"Error selecting best assertions: {str(e)}")
        selected_assertions = {}
    
    # Calculate overall alignment for the selected assertions
    try:
        if selected_assertions:
            overall_metrics = compute_overall_alignment(
                selected_assertions,
                evaluator.results,
                human_grades
            )
        else:
            # Default metrics when no assertions are selected
            overall_metrics = {
                "alignment_score": 0,
                "coverage": 0,
                "false_failure_rate": 0
            }
    except Exception as e:
        st.error(f"Error computing overall alignment: {str(e)}")
        # Default metrics when computation fails
        overall_metrics = {
            "alignment_score": 0,
            "coverage": 0,
            "false_failure_rate": 0
        }
    
    # Create a dashboard-like layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Overall Alignment",
            f"{format_percentage(overall_metrics['alignment_score'])}",
            help="How well the selected assertions match your grades (harmonic mean of coverage and 1-FFR)"
        )
    
    with col2:
        st.metric(
            "Coverage",
            f"{format_percentage(overall_metrics['coverage'])}",
            help="Percentage of human-labeled 'bad' outputs that are correctly flagged"
        )
    
    with col3:
        st.metric(
            "False Failure Rate",
            f"{format_percentage(overall_metrics['false_failure_rate'])}",
            help="Percentage of human-labeled 'good' outputs that are incorrectly flagged"
        )
    
    # Show FFR threshold slider
    ffr_threshold = st.slider(
        "False Failure Rate Threshold",
        min_value=0.0,
        max_value=0.5,
        value=ffr_threshold,
        step=0.05,
        format="%0.2f",
        help="Maximum acceptable False Failure Rate for selecting assertions"
    )
    
    if ffr_threshold != DEFAULT_FALSE_FAILURE_RATE_THRESHOLD:
        # Recompute with the new threshold
        try:
            selected_assertions = select_best_assertions(
                evaluator,
                ffr_threshold=ffr_threshold
            )
            
            if selected_assertions:
                overall_metrics = compute_overall_alignment(
                    selected_assertions,
                    evaluator.results,
                    human_grades
                )
            else:
                # Default metrics when no assertions are selected
                overall_metrics = {
                    "alignment_score": 0,
                    "coverage": 0,
                    "false_failure_rate": 0
                }
            
            st.info(f"Assertions re-selected with FFR threshold: {ffr_threshold}")
        except Exception as e:
            st.error(f"Error recalculating with new threshold: {str(e)}")
    
    # Explore the data in tabbed sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Selected Assertions",
        "🧠 Variant Analysis", 
        "🔄 Criteria Evolution",
        "🔍 Detailed Metrics"
    ])
    
    with tab1:
        st.subheader("Selected Assertions")
        
        st.markdown("""
        These assertions have been selected as the best representatives of your grading preferences.
        Each assertion is chosen to maximize alignment with your human judgments while staying below
        the False Failure Rate threshold.
        """)
        
        # Create a summary of selected assertions
        if metrics is None:
            metrics = {}
        selected_summary = create_selected_assertions_summary(evaluator, selected_assertions, metrics)
        
        if selected_summary:
            # Display summary table
            summary_df = pd.DataFrame(selected_summary)
            st.dataframe(
                summary_df,
                column_config={
                    "criterion": st.column_config.TextColumn("Criterion"),
                    "variant_name": st.column_config.TextColumn("Variant"),
                    "description": st.column_config.TextColumn("Approach"),
                    "coverage": st.column_config.ProgressColumn(
                        "Coverage",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "false_failure_rate": st.column_config.ProgressColumn(
                        "FFR",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "alignment_score": st.column_config.ProgressColumn(
                        "Alignment",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "total_variants": st.column_config.NumberColumn("Total Variants")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Show the assertion implementations
            for assertion in selected_summary:
                with st.expander(f"**{assertion['criterion']}** - {assertion['variant_name']}"):
                    st.markdown(f"**Description:** {assertion['description']}")
                    st.markdown("**Implementation:**")
                    st.code(assertion['implementation'], language="text")
            
            # Show alignment heatmap
            st.subheader("Alignment Heatmap")
            st.markdown("""
            This heatmap shows how the selected assertions align with your human grades.
            Each row represents a response, and each column represents a criterion.
            - **Green**: Assertion passes (good)
            - **Red**: Assertion fails (bad)
            """)
            
            # Create and display the heatmap
            heatmap_fig = create_assertion_heatmap(evaluator, selected_assertions)
            st.pyplot(heatmap_fig)
        
        else:
            st.warning("No assertions were selected. Try adjusting the FFR threshold.")
    
    with tab2:
        st.subheader("Assertion Variant Analysis")
        
        st.markdown("""
        For each criterion, multiple assertion variants were generated using different prompting strategies.
        This analysis shows how each variant performed and which one was selected as the best match for your preferences.
        """)
        
        # Create and display variant comparison chart
        if metrics is None:
            metrics = {}
        variant_fig = plot_variant_comparison(evaluator, metrics)
        st.pyplot(variant_fig)
        
        # Show detailed variant data
        variant_data = []
        for criterion_name, criterion_metrics in metrics.items():
            if "variant_scores" in criterion_metrics:
                for variant_id, score in criterion_metrics["variant_scores"].items():
                    # Get variant info
                    variant_info = next((a for a in evaluator.assertions.get(criterion_name, []) if a["id"] == variant_id), {})
                    
                    variant_data.append({
                        "criterion": criterion_name,
                        "variant_id": variant_id,
                        "variant_name": variant_info.get("variant_name", "Unknown"),
                        "description": variant_info.get("description", "Unknown"),
                        "coverage": criterion_metrics.get("variant_coverage", {}).get(variant_id, 0),
                        "false_failure_rate": criterion_metrics.get("variant_ffr", {}).get(variant_id, 0),
                        "alignment_score": score,
                        "selected": variant_id == criterion_metrics.get("best_assertion_id", "")
                    })
        
        if variant_data:
            # Convert to DataFrame
            variant_df = pd.DataFrame(variant_data)
            
            # Display as interactive table
            st.dataframe(
                variant_df,
                column_config={
                    "criterion": st.column_config.TextColumn("Criterion"),
                    "variant_name": st.column_config.TextColumn("Variant"),
                    "description": st.column_config.TextColumn("Approach"),
                    "coverage": st.column_config.ProgressColumn(
                        "Coverage",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "false_failure_rate": st.column_config.ProgressColumn(
                        "FFR",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "alignment_score": st.column_config.ProgressColumn(
                        "Alignment",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "selected": st.column_config.CheckboxColumn("Selected")
                },
                hide_index=True,
                use_container_width=True
            )
        
        else:
            st.warning("No variant data available.")
    
    with tab3:
        st.subheader("Criteria Evolution")
        
        evolution_summary = create_criteria_evolution_summary(evaluator)
        
        if evolution_summary["versions"] > 0:
            # Show summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Versions",
                    evolution_summary["versions"]
                )
            
            with col2:
                st.metric(
                    "Initial Criteria",
                    evolution_summary["criteria_count_initial"]
                )
            
            with col3:
                delta = evolution_summary["criteria_count_final"] - evolution_summary["criteria_count_initial"]
                st.metric(
                    "Final Criteria",
                    evolution_summary["criteria_count_final"],
                    delta=delta
                )
            
            # Show changes between versions
            if evolution_summary["changes"]:
                st.markdown("### Criteria Changes Over Time")
                
                for change in evolution_summary["changes"]:
                    # Only show versions with actual changes
                    if change["added_count"] > 0 or change["removed_count"] > 0 or change["modified_count"] > 0:
                        with st.expander(f"Version {change['from_version']} → {change['to_version']}: {change['source']}"):
                            if change["added_count"] > 0:
                                st.markdown(f"**Added ({change['added_count']}):** {', '.join(change['added'])}")
                            
                            if change["removed_count"] > 0:
                                st.markdown(f"**Removed ({change['removed_count']}):** {', '.join(change['removed'])}")
                            
                            if change["modified_count"] > 0:
                                st.markdown(f"**Modified ({change['modified_count']}):** {', '.join(change['modified'])}")
            
            # Show criteria history versions
            if "criteria_history" in st.session_state:
                st.markdown("### Full Criteria History")
                
                for i, version in enumerate(st.session_state.criteria_history):
                    with st.expander(f"Version {i+1}: {version['source']}"):
                        for j, criterion in enumerate(version["criteria"]):
                            st.markdown(f"**{j+1}. {criterion['name']}**")
                            st.markdown(f"{criterion['description']}")
        
        else:
            st.info("No criteria evolution history available.")
    
    with tab4:
        st.subheader("Detailed Metrics")
        
        # Display per-criterion metrics
        metrics_data = []
        
        if metrics is None:
            metrics = {}
            
        for criterion_id, criterion_metrics in metrics.items():
            if criterion_id in selected_assertions:
                metrics_data.append({
                    "name": criterion_id,
                    "metrics": {
                        "coverage": criterion_metrics.get("coverage", 0),
                        "false_failure_rate": criterion_metrics.get("ffr", 0),
                        "alignment_score": criterion_metrics.get("alignment", 0)
                    },
                    "confusion_matrix": criterion_metrics.get("confusion_matrix", {})
                })

                # Ensure confusion_matrix is a dictionary and has default values if missing
                confusion_matrix = criterion_metrics.get("confusion_matrix", {})
                if not isinstance(confusion_matrix, dict):
                    confusion_matrix = {}
                
                metrics_data.append({
                    "name": criterion_id,
                    "metrics": {
                        "coverage": criterion_metrics.get("coverage", 0),
                        "false_failure_rate": criterion_metrics.get("ffr", 0),
                        "alignment_score": criterion_metrics.get("alignment", 0)
                    },
                    "confusion_matrix": confusion_matrix
                })
        
        # Create a list to remove duplicate entries (caused by the double append)
        metrics_data_fixed = []
        seen_names = set()
        
        for item in metrics_data:
            if item["name"] not in seen_names:
                seen_names.add(item["name"])
                metrics_data_fixed.append(item)
        
        # Replace metrics_data with the fixed version without duplicates
        metrics_data = metrics_data_fixed
        
        if metrics_data:
            # Create metrics comparison chart
            metrics_fig = plot_metrics_comparison(metrics_data)
            st.pyplot(metrics_fig)
            
            # Display metrics as a table
            st.subheader("Metrics Table")
            
            # Create a DataFrame for the metrics with proper default values
            metrics_df = pd.DataFrame([
                {
                    "Criterion": m["name"],
                    "Coverage": m["metrics"].get("coverage", 0),
                    "FFR": m["metrics"].get("false_failure_rate", 0),
                    "Alignment": m["metrics"].get("alignment_score", 0),
                    "TP": m["confusion_matrix"].get("true_positive", 0),
                    "TN": m["confusion_matrix"].get("true_negative", 0),
                    "FP": m["confusion_matrix"].get("false_positive", 0),
                    "FN": m["confusion_matrix"].get("false_negative", 0)
                }
                for m in metrics_data
            ])
            
            # Display the table
            st.dataframe(
                metrics_df,
                column_config={
                    "Criterion": st.column_config.TextColumn("Criterion"),
                    "Coverage": st.column_config.ProgressColumn(
                        "Coverage",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "FFR": st.column_config.ProgressColumn(
                        "False Failure Rate",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Alignment": st.column_config.ProgressColumn(
                        "Alignment",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "TP": st.column_config.NumberColumn("True Positive"),
                    "TN": st.column_config.NumberColumn("True Negative"),
                    "FP": st.column_config.NumberColumn("False Positive"),
                    "FN": st.column_config.NumberColumn("False Negative")
                },
                hide_index=True,
                use_container_width=True
            )
        
        else:
            st.warning("No metrics data available.")
    
    # Export functionality
    if download_enabled:
        st.subheader("Export Selected Assertions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as JSON
            if st.button("Download as JSON", type="primary"):
                # Create a JSON export of the selected assertions
                export_data = {
                    "criteria": evaluator.criteria if hasattr(evaluator, "criteria") else [],
                    "selected_assertions": selected_assertions,
                    "metrics": {
                        "overall": overall_metrics,
                        "per_criterion": metrics
                    }
                }
                
                # Convert to JSON string
                json_str = json.dumps(export_data, indent=2)
                
                # Create a download link
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="selected_assertions.json",
                    mime="application/json"
                )
        
        with col2:
            # Export as Python code
            if st.button("Export as Python Code", type="primary"):
                # Create Python code for the selected assertions
                code = generate_python_code(selected_assertions, evaluator)
                
                # Create a download link
                st.download_button(
                    label="Download Python Code",
                    data=code,
                    file_name="selected_assertions.py",
                    mime="text/plain"
                )
    
    return selected_assertions, metrics


def generate_python_code(selected_assertions: Dict[str, Dict], evaluator) -> str:
    """
    Generate Python code for the selected assertions.
    
    Args:
        selected_assertions: Dictionary of selected assertions
        evaluator: Evaluator instance
        
    Returns:
        Python code as a string
    """
    code = """\"\"\"
Selected assertions for LLM response evaluation.

This file was automatically generated by LLM Evaluator.
\"\"\"

from typing import Dict, List, Optional, Union, Any


def evaluate_response(response: str, question_data: Optional[Dict] = None) -> Dict[str, Any]:
    \"\"\"
    Evaluate an LLM response using the selected assertions.
    
    Args:
        response: The LLM response to evaluate
        question_data: Optional question data for context-aware assertions
        
    Returns:
        Dictionary with evaluation results
    \"\"\"
    results = {}
    
    # Run each assertion
    assertion_results = {}
"""
    
    # Add each assertion as a function
    for i, (criterion_id, assertion) in enumerate(selected_assertions.items()):
        # Find the full assertion info
        assertion_info = next(
            (a for a in evaluator.assertions.get(criterion_id, []) if a["id"] == assertion["id"]),
            {}
        )
        
        # Create a function name
        function_name = f"check_{assertion['id'].replace('-', '_')}"
        
        # Add the function
        code += f"""
    # Assertion for: {criterion_id}
    assertion_results["{criterion_id}"] = check_{i+1}(response, question_data)
"""
    
    # Add the return statement
    code += """
    # Compute overall result
    passing_count = sum(1 for result in assertion_results.values() if result["passes"])
    total_count = len(assertion_results)
    
    results["assertions"] = assertion_results
    results["overall"] = {
        "passes": passing_count / total_count if total_count > 0 else 0,
        "pass_count": passing_count,
        "total_count": total_count
    }
    
    return results

"""
    
    # Add each assertion function
    for i, (criterion_id, assertion) in enumerate(selected_assertions.items()):
        # Find the full assertion info
        assertion_info = next(
            (a for a in evaluator.assertions.get(criterion_id, []) if a["id"] == assertion["id"]),
            {}
        )
        
        implementation = assertion_info.get("implementation", "")
        description = assertion_info.get("description", "")
        
        # Add the function
        code += f"""
def check_{i+1}(response: str, question_data: Optional[Dict] = None) -> Dict[str, Any]:
    \"\"\"
    Check if the response meets the criterion: {criterion_id}
    
    Approach: {description}
    
    Args:
        response: The LLM response to evaluate
        question_data: Optional question data for context
        
    Returns:
        Dictionary with assertion result
    \"\"\"
    # Import necessary modules
    import openai
    
    # Implementation
    prompt = \"\"\"
{implementation}
\"\"\"
    
    # Format the prompt with the response and question data if available
    if question_data and "{{question}}" in prompt:
        formatted_prompt = prompt.format(response=response, question=question_data.get("question", ""))
    else:
        formatted_prompt = prompt.format(response=response)
    
    # Call the LLM (this is a placeholder - replace with your actual LLM call)
    # In a real implementation, you would call your LLM API here
    try:
        # Example OpenAI call (replace with your actual implementation)
        result = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {{"role": "system", "content": "You are an evaluator assistant."}},
                {{"role": "user", "content": formatted_prompt}}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        # Parse the result (assuming the response contains 'PASS' or 'FAIL')
        result_text = result.choices[0].message.content
        passes = "PASS" in result_text.upper()
        
        return {{
            "passes": passes,
            "explanation": result_text,
            "error": None
        }}
    except Exception as e:
        return {{
            "passes": False,
            "explanation": None,
            "error": str(e)
        }}
"""
    
    return code 