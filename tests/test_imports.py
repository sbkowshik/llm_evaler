"""Test that all package modules can be imported."""

import pytest


def test_import_config():
    """Test importing the config module."""
    try:
        from llm_evaler.src import config
        assert hasattr(config, "PROMPT_TEMPLATE")
    except ImportError as e:
        pytest.fail(f"Import error: {str(e)}")


def test_import_data_loader():
    """Test importing the data_loader module."""
    try:
        from llm_evaler.src import data_loader
        assert hasattr(data_loader, "load_mmlu_dataset")
    except ImportError as e:
        pytest.fail(f"Import error: {str(e)}")


def test_import_llm_client():
    """Test importing the llm_client module."""
    try:
        from llm_evaler.src import llm_client
        assert hasattr(llm_client, "generate_responses")
    except ImportError as e:
        pytest.fail(f"Import error: {str(e)}")


def test_import_criteria_generator():
    """Test importing the criteria_generator module."""
    try:
        from llm_evaler.src import criteria_generator
        assert hasattr(criteria_generator, "generate_criteria_with_llm")
    except ImportError as e:
        pytest.fail(f"Import error: {str(e)}")


def test_import_assertion_generator():
    """Test importing the assertion_generator module."""
    try:
        from llm_evaler.src import assertion_generator
        assert hasattr(assertion_generator, "generate_assertions")
    except ImportError as e:
        pytest.fail(f"Import error: {str(e)}")


def test_import_evaluator():
    """Test importing the evaluator module."""
    try:
        from llm_evaler.src import evaluator
        assert hasattr(evaluator, "Evaluator")
    except ImportError as e:
        pytest.fail(f"Import error: {str(e)}")


def test_import_alignment():
    """Test importing the alignment module."""
    try:
        from llm_evaler.src import alignment
        assert hasattr(alignment, "compute_alignment_metrics")
    except ImportError as e:
        pytest.fail(f"Import error: {str(e)}")


def test_import_utils():
    """Test importing the utils module."""
    try:
        from llm_evaler.src import utils
        assert hasattr(utils, "sample_for_grading")
    except ImportError as e:
        pytest.fail(f"Import error: {str(e)}")


def test_import_app():
    """Test importing the main app module."""
    try:
        from llm_evaler.app import app
        assert hasattr(app, "main")
    except ImportError as e:
        pytest.fail(f"Import error: {str(e)}")


def test_import_components():
    """Test importing the UI components modules."""
    try:
        from llm_evaler.app.components import (
            prompt_node,
            criteria_wizard,
            grading_interface,
            report_card,
        )
        assert hasattr(prompt_node, "prompt_node")
        assert hasattr(criteria_wizard, "criteria_wizard")
        assert hasattr(grading_interface, "grading_interface")
        assert hasattr(report_card, "report_card")
    except ImportError as e:
        pytest.fail(f"Import error: {str(e)}") 