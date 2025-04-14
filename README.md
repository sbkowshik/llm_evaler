# LLM Evaler: EvalGen Implementation

A framework for evaluating LLM outputs with evaluations that align with human preferences.

## Overview

LLM Evaler is an implementation of the EvalGen methodology, designed to create evaluations of LLM outputs that align with human preferences. It addresses the challenge of validating LLM-generated content by combining automated assertions with human feedback in an iterative, mixed-initiative workflow.

The system supports:
- Generating evaluation criteria suggestions
- Creating multiple assertion variants with different prompting strategies
- Collecting human feedback through binary grading
- Aligning assertions with human preferences
- Supporting criteria drift and refinement
- Transparent reporting of alignment metrics

## Key Features

### 1. Mixed-Initiative Criteria Generation
- Automatic suggestion of evaluation criteria based on task requirements
- Support for manual refinement and criteria evolution
- Tracking of criteria drift based on human observations during grading

### 2. Multiple Assertion Variants
- Generation of diverse LLM-based assertion implementations
- Different prompting strategies to evaluate the same criteria
- Comparative analysis to identify which variant best aligns with human judgments

### 3. Human-in-the-Loop Grading
- Simple thumbs-up/thumbs-down grading interface
- Capture of criteria drift notes during grading
- Intelligent sampling of examples to maximize information gain

### 4. Alignment Metrics
- Coverage: How well assertions identify "bad" outputs
- False Failure Rate: How often assertions incorrectly flag "good" outputs
- Alignment Score: Harmonic mean of coverage and (1-FFR)
- Transparent metrics for each criterion and assertion variant

### 5. Iterative Workflow
- Support for evolving criteria definitions
- Real-time updates to alignment metrics
- Criteria history tracking and visualization

### 6. Comprehensive Report Card
- Detailed visualization of assertion performance
- Variant comparison charts
- Criteria evolution history
- Exportable assertions for deployment

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

## Acknowledgments

- Based on the EvalGen methodology
- Utilizes OpenAI's GPT models for assertion generation and evaluation
- Built with Streamlit for the user interface 