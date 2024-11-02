# Language Model Introspection Experiment

This project extends the experiments from ["Looking Inward: Language Models Can Learn About Themselves by Introspection"](https://arxiv.org/abs/2410.13787) by Binder et al. (2024), exploring whether models can achieve introspection through in-context learning rather than fine-tuning.

## Overview

While the original paper demonstrated that language models can learn to predict their own behavior through fine-tuning, this project investigates whether similar capabilities can be achieved using few-shot in-context learning.

## Project Structure

- `utils.py`: Core utility functions for data handling and model interactions
- `collect_original_responses.py`: Collects baseline responses from the model
- `collect_hypothetical_responses.py`: Implements few-shot learning to collect the model's predictions about its own behavior
- `evaluate.py`: Evaluates prediction accuracy across different numbers of shots and generates visualization

## Setup

Configure your model API keys in environment variables:
```bash
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
```

## Usage

1. Collect original responses:
```bash
python collect_original_responses.py
```

2. Collect hypothetical responses:
```bash
python collect_hypothetical_responses.py
```

3. Evaluate results:
```bash
python evaluate.py
```

## References

Binder, F. J., Chua, J., Korbak, T., Sleight, H., Hughes, J., Long, R., Perez, E., Turpin, M., & Evans, O. (2024). Looking Inward: Language Models Can Learn About Themselves by Introspection. arXiv preprint arXiv:2410.13787.
