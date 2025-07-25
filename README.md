# Aspect-Sentiment Quad Prediction with Distilled Large Language Models

This repository contains the source code for the paper "Aspect-Sentiment Quad Prediction with Distilled Large Language Models", accepted at RANLP 2025.

## Introduction

This project explores two primary methodologies for Aspect-Based Sentiment Analysis (ABSA) quad prediction:

1.  **Chain-of-Thought (CoT) with Agents:** This approach decomposes the ABSA task into a sequence of smaller, more manageable steps. Each step is handled by a specialized "agent" that focuses on a specific element of the quad (aspect, sentiment, or category). The project dynamically permutes the order of these agents to identify the most effective processing chain.

2.  **In-Context Learning (ICL):** This is a few-shot learning approach where the model is provided with a small number of examples of the task within the prompt to guide its predictions.

The system is designed to be model-agnostic, supporting various Large Language Models (LLMs) from providers like OpenAI, Google, and Anthropic, as well as local models like Qwen.

## Project Structure

```
.Aspect-Sentiment Quad Prediction with Distilled Large Language Models/
├── config/                     # Configuration files for models and datasets
│   ├── api_keys.toml.template  # Template for API keys
│   └── *.toml                    # Model and dataset-specific configurations
├── data/
│   ├── raw/                      # Raw datasets
│   └── processed/                # Processed data and experiment results
├── src/
│   ├── main_gemini.py            # Main script for running experiments with Gemini models
│   ├── main_gpt.py               # Main script for running experiments with GPT models
│   ├── main_qwen.py              # Main script for running experiments with Qwen models
│   ├── chainer.py                # Implements the Chain-of-Thought (CoT) with agents
│   ├── icl.py                    # Implements the In-Context Learning (ICL)
│   ├── models.py                 # Unified interface for interacting with different LLMs
│   ├── elements.py               # Defines the components of the CoT chain (Aspects, Sentiments, Categories)
│   └── ...
├── src_aux/
│   ├── dataset_statistics.py     # Calculates statistics about the datasets
│   ├── eval_asqp.py              # Evaluates the performance of the models
│   ├── extract_categories.py     # Extracts categories from the datasets
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API keys:**

    *   Rename `config/api_keys.toml.template` to `config/api_keys.toml`.
    *   Open `config/api_keys.toml` and fill in your API keys for the services you want to use (e.g., OpenAI, Google Vertex AI).

## Running the Experiments

The main entry points for running the experiments are the `src/main_*.py` files. You can run the experiments for a specific model by executing the corresponding script:

*   **For Gemini models:**

    ```bash
    python src/main_gemini.py
    ```

*   **For GPT models:**

    ```bash
    python src/main_gpt.py
    ```

*   **For Qwen models:**

    ```bash
    python src/main_qwen.py
    ```

You can customize the experiments by modifying the main block in each of these files. For example, you can change the list of datasets, models, or the number of examples for ICL.

## Citation

If you use this code in your research, please cite our paper:

```
@inproceedings{your-name-ranlp-2025,
    title = {Aspect-Sentiment Quad Prediction with Distilled Large Language Models},
    author = {Your Name(s)},
    booktitle = {Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2025)},
    year = {2025}
}
```
