# KotlinSage

A fine-tuned language model for Kotlin development assistance, built using Phi-2 with LoRA (Low-Rank Adaptation).

## Overview

This project fine-tunes Microsoft's Phi-2 model on Kotlin Q&A data to create an AI assistant that can help with Kotlin/Android development questions.

### What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that adds small "adapter" matrices to a pre-trained model instead of updating all weights. This:
- Reduces GPU memory usage (~20GB → ~6GB)
- Keeps the original model weights intact
- Trains only ~0.3% of total parameters

## Project Structure

```
KotlinSage/
├── data/                    # Training data
│   └── kotlin_qa.jsonl      # Kotlin Q&A dataset (47 samples)
├── output/                  # Model checkpoints
│   └── final/              # Final trained model
├── scripts/                # Python scripts
│   ├── download_dataset.py # Download Kotlin Q&A from HuggingFace
│   ├── train.py           # Fine-tuning script with documentation
│   ├── evaluate.py        # Perplexity evaluation script
│   └── inference.py       # Test the model with questions
├── .gitignore
├── pyproject.toml         # UV project config
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.13+
- NVIDIA GPU with CUDA 13.x
- 16GB+ VRAM recommended

### Setup

```bash
# Clone and setup
cd ~/Projects/KotlinSage

# Install dependencies (creates .venv automatically)
uv venv
source .venv/bin/activate

# Download Kotlin Q&A dataset
python scripts/download_dataset.py
```

### Training

```bash
# Fine-tune the model (10 epochs, ~2 min on RTX 5080)
python scripts/train.py

# Output: Trained model saved to output/final/
```

### Evaluation

```bash
# Measure improvement (perplexity comparison)
python scripts/evaluate.py

# Example output:
# Base model perplexity:      8.32
# Fine-tuned perplexity:       5.66
# Improvement:                32.0%
# ✓ Model improved after fine-tuning!
```

### Inference

```bash
# Test with Kotlin questions
python scripts/inference.py output/final

# Example:
# Q: How do I create a singleton in Kotlin?
# A: object MySingleton {
#       fun hello() = "Hello!"
#    }
```

## Configuration

Key hyperparameters in `scripts/train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `microsoft/Phi-2` | Base model to fine-tune |
| `MAX_LEN` | 512 | Max tokens per sample |
| `LORA_R` | 16 | LoRA rank (higher = more capacity) |
| `num_train_epochs` | 10 | Training iterations |
| `learning_rate` | 3e-4 | LoRA typical: 3e-4, full fine-tune: 1e-5 |
| `per_device_train_batch_size` | 1 | Batch size per GPU |

## Dataset

We use the **JetBrains/Kotlin_QA** dataset from HuggingFace, containing Kotlin Q&A pairs similar to Stack Overflow questions.

- **Size**: 47 samples (small for demonstration)
- **Format**: JSONL with `Question` and `Answer` fields

To improve results, add more data:
1. Combine with other Kotlin datasets (search HuggingFace)
2. Scrape Kotlin forum Q&A
3. Add your own Android/Kotlin examples

## Hardware Requirements

| Model | VRAM Needed |
|-------|-------------|
| Phi-2 (2.7B) | ~8GB |
| TinyLlama (1.1B) | ~4GB |
| Qwen2.5-0.5B | ~3GB |

Tested on: NVIDIA RTX 5080 (16GB)

## Results

| Metric | Base Model | Fine-tuned | Improvement |
|--------|------------|------------|-------------|
| Perplexity | 8.32 | 5.66 | 32% |
| Training Loss | - | 1.19 | - |

The model generates Kotlin-specific code and answers after fine-tuning.

## Next Steps

To improve the model further:

1. **More Data**: 47 samples is very small; aim for 1000+
2. **Larger Model**: Try Qwen2.5-1.5B or Phi-3-mini
3. **Hyperparameter Tuning**: Adjust learning rate, LoRA rank
4. **More Epochs**: Try 20-50 epochs
5. **Better Prompts**: Refine the prompt format

## Acknowledgments

- [microsoft/Phi-2](https://huggingface.co/microsoft/Phi-2) - Base model
- [JetBrains/Kotlin_QA](https://huggingface.co/datasets/JetBrains/Kotlin_QA) - Dataset
- [PEFT](https://github.com/huggingface/peft) - LoRA implementation