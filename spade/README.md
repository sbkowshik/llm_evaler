# SPADE Implementation

This folder contains an implementation of SPADE (System for Prompt Analysis and Delta-based Evaluation) for automated evaluation of LLM outputs without human input, as described by Shankar et al.

## Overview

SPADE automatically generates and selects a minimal set of assertions (evaluation functions) to evaluate LLM outputs without requiring human input during the process. The workflow includes:

1. **Input Prompt and LLM Outputs**: SPADE takes as input a prompt template and a set of LLM-generated outputs.

2. **Default Criteria**: SPADE uses the default criteria defined in config.py instead of generating criteria.

3. **Candidate Assertion Generation**: For each criterion, SPADE generates multiple candidate LLM-based assertions that can evaluate whether an output passes or fails the criterion.

4. **Assertion Execution**: SPADE runs the candidate assertions on LLM outputs to determine pass/fail rates and selectivity.

5. **Optimization for Minimal Assertion Set**: SPADE selects a minimal set of assertions that covers all the criteria while meeting a defined false failure rate (FFR) threshold.

6. **Output Assertion Set**: The final set of assertions can be used to automatically evaluate future LLM outputs.

## Key Features

- **Fully Automated**: SPADE operates end-to-end without human input, unlike EvalGen which requires human feedback.
- **Focus on Minimality**: SPADE aims to reduce the number of assertions while still covering all criteria.
- **LLM-based Evaluation**: All assertions use LLM-based evaluation rather than code-based evaluation.
- **Default Criteria**: Uses the predefined criteria from config.py without generating new ones.

## Usage

To run SPADE on the LLM responses:

```bash
python -m spade.main --data-file llm_evaler/data/llm_responses.json --sample-size 50
```

### Parameters

- `--data-file`: Path to the JSON file containing LLM responses (default: `llm_evaler/data/llm_responses.json`)
- `--sample-size`: Number of responses to evaluate (default: 50)
- `--ffr-threshold`: Maximum false failure rate threshold (default: 0.4 or 40%)
- `--output-file`: Path to save results (default: `spade/results.json`)

## Comparison with EvalGen

This implementation is designed to compare with EvalGen. Key differences include:

1. EvalGen involves humans in criteria selection and output grading, while SPADE is fully automated.
2. SPADE aims for a minimal assertion set, while EvalGen focuses on alignment with human preferences.
3. SPADE may include more assertions to compensate for the lack of human feedback.

The comparison helps understand the trade-offs between fully automated approaches (SPADE) and approaches that incorporate human feedback (EvalGen).

## Implementation Details

- `src/config.py`: Configuration settings for SPADE
- `src/criteria_generator.py`: Contains the default criteria
- `src/assertion_generator.py`: Generation of LLM-based assertions
- `src/optimizer.py`: Selection of the minimal assertion set
- `src/spade.py`: Main SPADE implementation
- `main.py`: Script to run SPADE and compare with EvalGen

## Default Criteria

The implementation uses the following default criteria defined in config.py:

1. **Concise explanation**: Response is concise and to the point
2. **Straightforward explanation**: Explanation is straightforward and easy to understand

## Results

The results of running SPADE are saved to `spade/results.json`, which includes:

- Default criteria used
- Generated assertions
- Selected assertions (minimal set)
- Metrics for evaluating the performance of the selected assertion set
- Alignment metrics for comparison with EvalGen (if human grades are available) 