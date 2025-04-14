# EvalGen-like System Implementation Plan for MMLU Dataset

## 1. Project Setup

### Dependencies
```
streamlit==1.38.0
datasets>=3.5.0
transformers>=4.51.0
openai>=1.72.0
matplotlib>=3.7.0
seaborn>=0.13.0
numpy>=1.23.0
pandas>=2.0.0
tqdm>=4.66.0
python-dotenv>=1.0.0
```

### Project Structure
```
llm_evaler/
├── data/                      # Data storage
│   └── mmlu_samples.json      # Sampled MMLU questions
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── data_loader.py         # Load and process MMLU data
│   ├── llm_client.py          # OpenAI API client
│   ├── criteria_generator.py  # Generate evaluation criteria
│   ├── assertion_generator.py # Generate candidate assertions
│   ├── evaluator.py           # Run assertions on LLM outputs
│   ├── alignment.py           # Compute alignment metrics
│   └── utils.py               # Utility functions
└── app/
    ├── __init__.py
    ├── components/            # Streamlit UI components
    │   ├── __init__.py
    │   ├── prompt_node.py     # Prompt configuration component
    │   ├── criteria_wizard.py # Criteria selection wizard
    │   ├── grading_interface.py # Output grading interface
    │   └── report_card.py     # Alignment report visualization
    └── app.py                 # Main Streamlit application
```

## 2. Data Preparation & LLM Pipeline Setup

### 2.1 Data Sampling
- Load the MMLU dataset from Hugging Face (`cais/mmlu`)
- Focus on the `college_computer_science_test` category
- Sample 50-100 questions for our evaluation

### 2.2 Prompt Template
```
Answer the following multiple-choice question. Choose the correct option and provide a brief explanation.  
Question: [question_text]  
Options:  
A) [option1]  
B) [option2]  
C) [option3]  
D) [option4]  
```

### 2.3 LLM Output Generation
- Use GPT-4o-mini to generate responses for the sampled questions
- Store responses in JSON format with question ID, question text, options, and LLM response

## 3. Criteria Generation

### 3.1 Automated Criteria Suggestion
Use GPT-4o-mini to suggest evaluation criteria based on the task. Initial criteria may include:

1. **Correct Option Selection**: Response correctly identifies the option letter (A, B, C, or D)
2. **Explanation Presence**: Response includes an explanation for the answer
3. **Explanation Quality**: Explanation is clear, logical, and supports the selected option
4. **Conciseness**: Response is brief and avoids unnecessary elaboration
5. **Format Adherence**: Response follows the expected format (option letter + explanation)
6. **No Hallucination**: Explanation doesn't include unsupported claims or incorrect information

### 3.2 User Refinement Interface
- Allow users to edit, add, or remove criteria throughout the process
- Support criteria drift by enabling ongoing refinement as users grade more outputs
- Provide a criteria wizard that helps users articulate evaluation needs
- Allow criteria to be deactivated rather than deleted to preserve history

## 4. Candidate Assertion Synthesis

### 4.1 LLM-Based Assertions
For each criterion, generate multiple variants of LLM-based assertions with different prompting strategies:

```python
# Example variants for Correct Option Selection criterion
def generate_assertion_variants(criterion: str, question_data: dict) -> List[Callable]:
    variants = []
    
    # Variant 1: Direct question approach
    def variant_1(response: str, question_data: dict) -> bool:
        prompt = f"""
        System: You are evaluating a response to a multiple-choice question. Determine if the response 
        correctly identifies the option letter (A, B, C, or D) that matches the ground truth.
        
        Question: {question_data['question']}
        Options:
        A) {question_data['options'][0]}
        B) {question_data['options'][1]}
        C) {question_data['options'][2]}
        D) {question_data['options'][3]}
        
        Ground truth answer: {question_data['answer']}
        
        Response to evaluate: {response}
        
        Does the response correctly identify the option letter that matches the ground truth?
        Respond with only "yes" or "no".
        """
        
        result = call_llm(prompt)
        return "yes" in result.lower()
    
    # Variant 2: Step-by-step reasoning approach
    def variant_2(response: str, question_data: dict) -> bool:
        prompt = f"""
        System: You are evaluating a response to a multiple-choice question. Follow these steps:
        1. Identify which option letter (A, B, C, or D) the response selected
        2. Compare it to the ground truth answer: {question_data['answer']}
        3. Determine if they match
        
        Question: {question_data['question']}
        Options:
        A) {question_data['options'][0]}
        B) {question_data['options'][1]}
        C) {question_data['options'][2]}
        D) {question_data['options'][3]}
        
        Response to evaluate: {response}
        
        First extract the selected option, then state whether it matches the ground truth.
        End with "yes" if correct or "no" if incorrect.
        """
        
        result = call_llm(prompt)
        return "yes" in result.lower()
    
    variants.append(variant_1)
    variants.append(variant_2)
    
    return variants
```

## 5. Human Grading Interface

### 5.1 Grading UI (Streamlit)
- Display LLM response with corresponding question and context
- Provide thumbs-up/thumbs-down buttons for binary grading
- Optional comment field for feedback on "bad" responses
- Show criteria being evaluated to help users understand what they're grading against
- Pagination controls to navigate through outputs

### 5.2 Sampling Strategy
- Implement alternating policy from EvalGen to select outputs for grading
- Mix of high/low confidence scores based on dynamically updated assertions
- Prioritize examples where candidate assertions disagree to maximize information gain

## 6. Assertion Alignment

### 6.1 Alignment Metrics
- **Coverage**: Percentage of human-labeled "bad" outputs that are correctly flagged by assertions
- **False Failure Rate (FFR)**: Percentage of human-labeled "good" outputs that are incorrectly flagged
- **Alignment Score**: Harmonic mean of coverage and (1 - FFR)

### 6.2 Assertion Selection Algorithm
- For each criterion, rank candidate assertion variants by alignment with human grades
- Update alignment metrics after each human grading decision
- Select the top assertion per criterion that meets a user-defined FFR threshold (e.g., ≤15%)
- Combine selected assertions into the final evaluation set
- Allow users to override assertion selection if needed

## 7. Visualization & Reporting

### 7.1 Report Card UI
- Per-criterion alignment metrics (coverage, FFR, alignment score)
- Confusion matrices to visualize misalignments
- Highlighting of specific examples where assertions and human grades diverge
- Overall evaluation performance visualization
- Transparency into assertion decisions for debugging and trust

### 7.2 Result Export
- Export evaluation results as CSV/JSON for further analysis
- Export selected assertions as Python code or API calls for deployment
- Option to save selected assertions for future use on similar tasks

## 8. Workflow Implementation

### Phase 1: Setup and Data Preparation
- Install dependencies
- Create initial project structure
- Load and sample MMLU data
- Generate LLM outputs with GPT-4o-mini

### Phase 2: Core Components
- Implement criteria generator with suggestions
- Implement assertion variant generator
- Create alignment metrics calculator
- Build basic Streamlit interface

### Phase 3: User Interface & Iterative Workflow
- Implement criteria wizard with support for criteria refinement
- Develop grading interface with real-time feedback
- Create report card visualization with transparency features
- Implement the iterative loop for criteria refinement, grading, and assertion selection
- Add support for saving and loading progress

### Phase 4: Testing and Refinement
- Test complete workflow with sample MMLU questions
- Refine UI based on user experience
- Optimize assertion generation and alignment calculation
- Conduct usability testing with ML practitioners

## 9. Success Metrics

- Successful implementation of EvalGen-like workflow for MMLU dataset
- Ability to generate sensible criteria and assertions for MCQA tasks
- Support for criteria drift and iterative refinement
- Alignment of selected assertions with human preferences (target: >80% alignment)
- Intuitive UI that enables efficient human grading and review 