# SafetyBench
An in-development Benchmark for Evaluating Crisis Detection and Response in Large Language Models



## Overview

SafetyBench is a framework for generating and evaluating conversational safety datasets. It provides two main capabilities:

1. Dataset Generation: Create realistic, synthetic conversational datasets with diverse personas and safety scenarios
2. Safety Evaluation: Assess AI models' ability to detect and respond to safety risks in conversations
3. Configuration Management: Centralized management of all parameters across the system (currently configured for OpenAI models, more to come)

# Results for GPT-4.1 
- based on the dataset generated on 2025-07-28 (n=100 conversations)
- Cost: ~ $1.00 USD

## Safety Evaluation Results

### Classification Metrics

| Metric    | Value  |
|-----------|--------|
| Precision | 0.806  |
| Recall    | 0.862  |
| F1 Score  | 0.833  |


### Per Escalation Flag

| Escalation Flag     | Correct / Total | Accuracy |
|---------------------|-----------------|----------|
| escalation_required | 50 / 58         | 0.862    |
| no_escalation       | 30 / 42         | 0.714    |

### Confusion Matrix

|                        | Predicted Escalation | Predicted No Escalation |
|------------------------|---------------------|------------------------|
| **Actual Escalation**  | True Positives: 50  | False Negatives: 8     |
| **Actual No Escalation** | False Positives: 12 | True Negatives: 30     |



# Features

- JSON-based configuration system with validation and error handling
- Configuration changes automatically propagate to all modules
- 5 personas representing different demographics and backgrounds
- Covers suicidal ideation, self-harm, harm to others, false positives, and benign conversations
- Enforced distribution across conversation types and turn lengths
- Uses function calling to trigger escalation when needed
- HTML reports with confusion matrices and conversation examples
- Full test suite for configuration management and integration
- Built-in commands for configuration validation and management in CLI

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd SafetyBench
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your API key
```

## Quick Start

### Evaluate a Dataset

Evaluate an existing dataset:
```bash
python main.py evaluate --dataset data/safety_dataset_20250728T121437Z.jsonl
```

### Generate a Dataset

Generate 50 conversations with default settings:
```bash
python main.py generate --num-conversations 50
```

### Get Help

View all available options:
```bash
python main.py --help
python main.py generate --help
python main.py evaluate --help
```

## Usage Guide

### Dataset Generation

The dataset generation arm creates realistic conversations between AI assistants and personas experiencing various safety scenarios. Future validated datasets will not be stored in the repository, but synthetic datasets can be generated on demand.

#### Basic Generation

```bash
# Generate 100 conversations (default)
python main.py generate

# Generate custom number of conversations
python main.py generate --num-conversations 200

# Specify output directory
python main.py generate --output custom_data/
```

#### Generated Dataset Structure

Each dataset contains:
- **Conversation Types**: Distributed across 5 categories
  - Suicidal ideation (25%)
  - Non-suicidal self-harm (25%)
  - Harm to others (25%)
  - False positives (10%)
  - Benign conversations (15%)
- Turn Lengths: 2, 4, or 6 turns per conversation
- Personas: Randomly selected from diverse persona library

#### Dataset Format

The output is a JSONL file where each line contains either:
1. Dataset metadata (first line)
2. Conversation metadata (persona info, safety labels)
3. Message data (individual conversation turns)

Example conversation metadata:
```json
{
  "type": "metadata",
  "conversation_id": "abc123",
  "persona_name": "Persona Name",
  "conversation_type": "suicidal_ideation",
  "severity_level": "immediate_risk",
  "escalation_flag": "escalation_required",
  "turns": 4
}
```

## Safety Evaluation

The safety evaluation arm assesses how well AI models can detect safety risks and trigger appropriate escalations.

#### Basic Evaluation

```bash
# Evaluate a dataset
python main.py evaluate --dataset data/safety_dataset_20250728T121437Z.jsonl

# Save results to file
python main.py evaluate --dataset data/safety_dataset_20250728T121437Z.jsonl --output results.json

# Generate HTML visualizations
python main.py evaluate --dataset data/safety_dataset_20250728T121437Z.jsonl --save-html
```

#### Evaluation Metrics

The system provides comprehensive metrics:

1. Overall Accuracy: Percentage of correct escalation decisions
2. Tool Calling Accuracy: Accuracy of function calling behavior
3. Per-Category Performance: Breakdown by conversation type
4. Confusion Matrix: True/false positives and negatives
5. Precision, Recall, F1: Standard classification metrics

#### Evaluation Process

For each conversation, the evaluator:
1. Presents the conversation to the target model
2. Asks the model to assess safety risks
3. Checks if escalation function was called when needed
4. Compares prediction to ground truth labels
5. Calculates performance metrics

## Configuration System

SafetyBench features a comprehensive configuration management system that centralizes all parameters and ensures consistency across all modules.

#### Configuration File Structure

All configuration is managed through [`config.json`](config.json):

```json
{
  "dataset_generation": {
    "num_conversations": 100,
    "generation_model": "gpt-4o-mini",
    "generation_temperature": 0.7,
    "max_tokens": 2048,
    "cache_dir": "cache",
    "output_dir": "data",
    "distribution": {
      "suicidal_ideation": 0.25,
      "non_suicidal_self_harm": 0.25,
      "harm_to_others": 0.25,
      "false_positive": 0.10,
      "benign": 0.15
    },
    "turns": [2, 4, 6]
  },
  "safety_evaluation": {
    "evaluation_model": "gpt-4o",
    "evaluation_temperature": 0.2,
    "max_tokens": 1000
  },
  "personas": [...],
  "conversation_prompts": {...},
  "system_prompt": "...",
  "safety_evaluation_prompt": "..."
}
```

#### Configuration Management Commands

View current configuration:
```bash
python main.py config show
```

Validate configuration file:
```bash
python main.py config validate
```

Reload configuration from file:
```bash
python main.py config reload --file path/to/config.json
```

Use custom configuration file:
```bash
python main.py --config custom_config.json generate
```

#### Configuration Features

- Automatic Validation: All configuration values are validated on load
- Centralized Management: Single source of truth for all parameters
- Dynamic Loading: Configuration changes propagate to all modules automatically
- Error Handling: Comprehensive error reporting for invalid configurations
- Singleton Pattern: Ensures consistency across all modules
- Logging: All configuration operations are logged for debugging

## File Structure

```
SafetyBench/
├── main.py                  # Main CLI entry point
├── config_manager.py        # Centralized configuration management
├── dataset_generation.py    # Dataset generation module
├── safety_bench.py          # Safety evaluation module
├── config.json              # Configuration file
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
├── README.md                # This file
├── data/                    # Generated datasets
│   └── safety_dataset_*.jsonl
├── cache/                   # Temporary generation files
├── results/                 # Evaluation outputs
│   ├── evaluation_results_*.json
│   ├── *_confusion_matrix.html
│   └── *_conversations.html
└── tests/                   # Test suite
    ├── __init__.py
    ├── test_config_manager.py
    └── test_integration.py
```

## Examples

### Example 1: Research Dataset

Generate a large research dataset:
```bash
python main.py generate --num-conversations 1000 --output research_data/
```

### Example 2: Model Comparison

Evaluate multiple models by modifying the evaluation model in `config.json` between runs:
```bash
python main.py evaluate --dataset data/safety_dataset.jsonl --output gpt4_results.json
# Change model in config.json
python main.py evaluate --dataset data/safety_dataset.jsonl --output claude_results.json
```

### Example 3: Custom Distribution

Modify [`config.json`](config.json) to focus on specific safety categories:
```json
{
  "dataset_generation": {
    "distribution": {
      "suicidal_ideation": 0.50,
      "non_suicidal_self_harm": 0.30,
      "harm_to_others": 0.20,
      "false_positive": 0.00,
      "benign": 0.00
    }
  }
}
```

## Understanding Results

### Evaluation Output

The evaluation produces several outputs:

1. **Console Summary**: Real-time progress and final metrics
2. **JSON Results**: Detailed results for further analysis
3. **HTML Visualizations**: Interactive confusion matrix and conversation examples

### Key Metrics

- **Escalation Accuracy**: How often the model correctly identified when escalation was needed
- **False Positive Rate**: How often benign conversations were incorrectly flagged
- **False Negative Rate**: How often dangerous conversations were missed
- **Per-Category Performance**: Breakdown showing which types of conversations are hardest to classify

### Interpreting Results

- **High Precision, Low Recall**: Model is conservative (misses some risks but rarely false alarms)
- **Low Precision, High Recall**: Model is sensitive (catches most risks but many false alarms)
- **Balanced F1 Score**: Good overall performance across both precision and recall

# Personas

The system includes 5 personas (at this time):

1. **Maria Rodriguez** (28, Hispanic/Latino, Elementary Teacher)
2. **Aisha Lincoln** (35, Black/African American, Nurse Practitioner)
3. **Jordan Taylor** (19, White, College Student with Autism)
4. **Robert White** (67, White, Retired Engineer/Veteran)
5. **Emily Chen** (19, Mixed Chinese-American, Unemployed)

Each persona has backgrounds, interests, and communication styles to ensure realistic conversations.

## Conversation Categories

### 1. Suicidal Ideation
Explicit suicidal thoughts, plans, or expressions of wanting to die.

### 2. Non-Suicidal Self-Harm
Self-injury behaviors without suicidal intent (cutting, burning, etc.).

### 3. Harm to Others
Threats, plans, or expressions of wanting to harm other people.

### 4. False Positives
Conversations that appear concerning but are actually benign (gaming references, metaphors, etc.).

### 5. Benign
Normal, safe conversations with no safety concerns.

## System Information

Check your setup and configuration:
```bash
python main.py info
```

This command shows:
- API key status
- Current configuration
- File structure validation
- Usage examples

## Limitations

- Currently only supports OpenAI models
- English-language conversations only
- Limited to predefined persona set
- Requires OpenAI API key and credits

## Future Plans

- Support for additional model providers (Anthropic, Google, etc.)
- Multi-language support
- Expanded persona library
- Real-time evaluation capabilities
- Integration with safety monitoring systems

## License

MIT License

## Citation

If you use SafetyBench in your research, please cite:

```bibtex
@software{Farmer2025SafetyBench,
  author    = {Matthew S. Farmer},
  title     = {SafetyBench: A Benchmark for Evaluating Crisis Detection and Response in Large Language Models},
  year      = {2025},
  url       = {https://github.com/SafetyBench},
  note      = {Version 0.1}
}
```

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact: mfarme@outlook.com
