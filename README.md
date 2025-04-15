# LLM Evaluation System

This repository implements the EvalGen approach for aligning automated LLM evaluations with human preferences, as described in the paper "EvalGen: From Criteria Prompts to Aligned Evaluations."

## System Architecture

The EvalGen system consists of three main components:

1. **Criteria Suggestion**: Uses GPT-4 to propose binary evaluation criteria in natural language (e.g., response length or tone).

2. **Candidate Assertion Synthesis and Execution**: Generates multiple candidate assertions (as code or LLM prompts) for each criterion, and executes them on LLM pipeline outputs.

3. **Grading Sampler**: Samples LLM outputs for the user to grade (thumbs up/down), then dynamically updates alignment metrics for each candidate assertion.

## Key Features

- **Confidence-based Sampling**: Intelligently samples LLM outputs for grading based on assertion selectivity and confidence scores.
- **Multiple Assertion Variants**: Generates and tests multiple implementations for each criterion to find the most aligned with user preferences.
- **Alignment Metrics**: Uses coverage (ability to catch bad outputs) and false failure rate (ability to not fail good outputs) to measure assertion quality.
- **Streaming Architecture**: Progressively updates all metrics as more grades are collected.

## How It Works

1. Users select evaluation criteria from suggestions or add their own.
2. The system asynchronously generates multiple candidate assertions for each criterion.
3. All assertions are executed on the LLM outputs, and selectivity estimates are updated.
4. The system samples LLM outputs for the user to grade, prioritizing potentially problematic outputs.
5. As the user grades, alignment metrics are updated to identify the best assertions.
6. The final set of assertions is selected based on their alignment with user grades.

## Implementation Details

The system is implemented with the following key components:

- `alignment.py`: Computes alignment metrics between assertions and human grades.
- `evaluator.py`: Runs assertions on LLM responses and stores results.
- `assertion_generator.py`: Generates candidate assertions (code and LLM-based).
- `utils.py`: Provides sampling strategies for human grading.

## Key Changes in this Implementation

This implementation differs from prior work in two main ways:

1. It uses a streaming architecture that progressively updates alignment metrics as more human grades are collected.
2. It employs a confidence-based sampling strategy that prioritizes outputs that are more likely to be of low quality.

## Usage

To use the system:

1. Run assertions on LLM outputs to generate initial evaluations.
2. Collect human grades on a subset of outputs using the grading interface.
3. The system automatically selects the most aligned assertions for each criterion.
4. Use the selected assertions to evaluate new LLM outputs with confidence that they align with human preferences.

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm_evaler.git
cd llm_evaler
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_api_key_here
```

### Running the Application

Start the Streamlit application:
```bash
python run.py
```

This will launch the web interface at http://localhost:8501 by default.

## Usage Workflow

### 1. Prompt Configuration
- Define the prompt for generating LLM responses
- Select a dataset (MMLU by default)
- Generate sample responses for evaluation

### 2. Criteria Definition
- Generate suggested criteria using LLM
- Edit, add, or remove criteria as needed
- Define criteria based on your quality standards

### 3. Assertion Generation
- For each criterion, generate multiple assertion variants
- Each variant uses a different prompting strategy
- Review generated assertions before proceeding

### 4. Response Grading
- Grade a subset of LLM responses (thumbs-up/thumbs-down)
- Provide feedback on criteria drift when noticed
- View real-time alignment metrics as you grade

### 5. Report Card
- Review selected assertions that best align with your judgments
- Explore variant performance and metrics
- Export selected assertions for deployment

## Data Flow

```
┌─────────────┐     ┌────────────┐     ┌─────────────┐     ┌────────────┐     ┌──────────┐
│    Prompt   │────▶│  Generate  │────▶│   Define    │────▶│  Generate  │────▶│   Grade  │
│Configuration│     │ Responses  │     │  Criteria   │     │ Assertions │     │Responses │
└─────────────┘     └────────────┘     └─────────────┘     └────────────┘     └────┬─────┘
                                             ▲                                      │
                                             │                                      │
                                             │                                      ▼
                                      ┌──────┴───────┐                        ┌──────────┐
                                      │   Criteria   │◀───────────────────────│  Report  │
                                      │Drift & Update│                        │   Card   │
                                      └──────────────┘                        └──────────┘
```

## Extending the System

### Adding New Datasets
Add new dataset loaders in `src/data_loader.py` and update the UI in `app/components/prompt_node.py`.

### Adding New Assertion Types
Extend the assertion generator in `src/assertion_generator.py` with new prompting strategies.

### Custom Evaluation Metrics
Add custom metrics in `src/alignment.py` and visualize them in `app/components/report_card.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


# 1.Why this paper is important and where it is limited
The paper tackles a critical issue in the LLM ecosystem—ensuring that automated evaluations (often conducted by other LLMs) align with human judgment. By combining automated LLM-generated evaluations with human feedback, the authors bridge the gap between efficiency and accuracy, reducing over-reliance on potentially flawed automated systems.

# 2.⁠ ⁠If given a chance, how and where you would advance your research
Automated learning through RLHF ^ Will be discussed in the call


