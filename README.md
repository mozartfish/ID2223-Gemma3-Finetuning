# ID2223-Gemma3-Finetuning
Finetuning [Gemma3 1B (text-only)](https://deepmind.google/models/gemma/gemma-3/) models on the finetome instruction dataset using data-centric and model-centric approaches

`Baseline` - [Model Repo](https://huggingface.co/mozartfish/Gemma3-FineTome10K-Baseline) | [Demo](https://huggingface.co/spaces/tanudin/lab2_docker_version)

`Data-Centric` - [Model Repo](https://huggingface.co/mozartfish/Gemma3-FineTome30K-data-centric) | [Demo](https://huggingface.co/spaces/tanudin/lab2_docker_version)

`Model-Centric` - [Model Repo](https://huggingface.co/mozartfish/Gemma3-FineTome10K-model-centric) | [Demo](https://huggingface.co/spaces/tanudin/lab2_docker_version)

# Gemma-3 Fine-Tuned LLM Evaluation App

A comprehensive Gradio-based web application for interacting with and evaluating fine-tuned Gemma-3 language models.

## Overview

This application provides an interactive interface to test and compare three versions of fine-tuned Gemma-3 models:
- **Baseline (10K)**: Trained on 10,000 samples
- **Model-Centric (10K)**: Trained on 10,000 samples with optimized hyperparameters
- **Data-Centric (30K)**: Trained on 30,000 high-quality samples

## Features

### 1. Chatbot
Interactive chat interface using the Data-Centric model. Have conversations and test the model's conversational abilities.

### 2. Text Rewriter
Transform and rewrite your text in different styles and tones. Choose from:
- **Gen-Z**: Casual slang, modern internet culture, relatable vibes
- **Formal Corporate**: Professional business language, polished and concise
- **Academic**: Scholarly discourse, intellectual rigor, educational depth
- **Humorous**: Witty jokes, wordplay, entertaining responses
- **Teacher Mode**: Patient explanations, step-by-step breakdowns, supportive guidance

This feature demonstrates the fine-tuned model's ability to adapt writing style through prompt engineering. Perfect for content creators, students, and professionals who need to adjust their message for different audiences.

### 3. Batch Model Comparison
Compare all three models using perplexity metrics. Requires text with at least 1024 tokens (~750-800 words). Includes pre-filled sample text for immediate testing.

### 4. Speed Benchmark
Measure and compare generation speed (tokens per second) across all three models. Helps identify which model is fastest for production use.

### 5. Task Testing
Evaluate models on specific tasks:
- **Math**: Calculations and numerical reasoning
- **Reasoning**: Logical thinking and problem-solving
- **Summarization**: Text condensation skills
- **Code**: Programming ability

## Technology Stack

- **Frontend**: Gradio
- **Backend**: Python 3.10
- **LLM Runtime**: llama.cpp
- **Models**: GGUF quantized models
- **Deployment**: Docker container


## Key Parameters

Adjust these sliders in the UI:
- **Temperature** (0-2): Controls randomness. Lower = more focused, Higher = more creative
- **Top-p** (0-1): Nucleus sampling. Controls diversity of outputs
- **Max Tokens** (32-1024): Maximum length of generated responses

## Evaluation Metrics

### Perplexity (Lower is Better)
Measures how well the model predicts text. Lower perplexity indicates better language modeling.

### Tokens/Second (Higher is Better)
Measures generation speed. Important for production deployment and user experience.

### Qualitative Comparison
Task testing allows manual evaluation of response quality, coherence, and accuracy.

## Finetuning Stategy 

### Baseline 

first 10K samples from Finetome instruction dataset 

- r = 8 
- Lora_alpha = 8 
- Lora_dropout = 0 
- Warmup_steps = 5 
- Num epochs = 3 
- Learning rate = 2e-4 
- Weight decay = 0.001 

### Data Centric

30K samples from Finetome instruction dataset. selected instructions 50,000-80,000 to ensure no overlap with training data used in baseline. 

- r = 8 
- Lora_alpha = 8 
- Lora_dropout = 0 
- Warmup_steps = 5 
- Num epochs = 1
- Learning rate = 2e-4 
- Weight decay = 0.001 

### Model Centric 

10K samples from Finetome instruction dataset but with optimized hyperparameters -> adjusted decay, peanlty, warmup steps
- r = 16
- Lora_alpha = 32 
- Lora_dropout = 0.05
- Warmup_steps = 50
- Num epochs = 3 
-  Learning rate = 1e-4
-  Weight decay = 0.01 
