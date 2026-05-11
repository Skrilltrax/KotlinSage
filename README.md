# KotlinSage

A fine-tuned language model for Kotlin development assistance, built using Qwen2.5-3B with LoRA (Low-Rank Adaptation).

## Overview

This project fine-tunes Qwen2.5-3B on Kotlin Q&A data to create an AI assistant that can help with Kotlin/Android development questions.

### What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that adds small "adapter" matrices to a pre-trained model instead of updating all weights. This:
- Reduces GPU memory usage (~20GB → ~6GB)
- Keeps the original model weights intact
- Trains only ~0.3% of total parameters

## Project Structure

```
KotlinSage/
├── data/                    # Training data (auto-generated)
├── output/                  # Model checkpoints
│   └── final/              # Final trained model
├── scripts/                # Python scripts
│   ├── download_dataset.py # Download & combine Kotlin datasets
│   ├── train.py           # Fine-tuning script
│   ├── evaluate.py        # Perplexity evaluation
│   ├── inference.py       # Test the model
│   └── utils.py           # Model-specific configs (Qwen, Phi, Llama)
├── ks                     # CLI entry point
├── pyproject.toml         # UV project config
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.13+
- NVIDIA GPU with CUDA 13.x
- 16GB+ VRAM recommended

### CLI Usage

```bash
cd ~/Projects/KotlinSage
source .venv/bin/activate

# Full pipeline (setup + train + eval + infer)
./ks all

# Individual commands
./ks setup    # Install deps & download data
./ks train    # Fine-tune model
./ks eval     # Evaluate perplexity
./ks infer    # Test with questions
```

### Manual Commands

```bash
# Setup
uv venv
source .venv/bin/activate
uv add torch transformers peft bitsandbytes accelerate datasets
python scripts/download_dataset.py

# Train
python scripts/train.py

# Evaluate
python scripts/evaluate.py

# Inference
python scripts/inference.py output/final
```

## Configuration

Key hyperparameters in `scripts/train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen2.5-3B` | Base model (supports Phi-2, Qwen, Llama) |
| `MAX_LEN` | 512 | Max tokens per sample |
| `LORA_R` | 16 | LoRA rank |
| `num_train_epochs` | 3 | Training iterations |
| `learning_rate` | 3e-4 | LoRA typical: 3e-4 |
| `per_device_train_batch_size` | 1 | Batch size per GPU |

### Switching Models

The codebase automatically detects and adapts prompts for different models. Just change `MODEL_NAME` in `train.py`:

```python
MODEL_NAME = "microsoft/Phi-2"   # Uses <|user|>/<|answer|> format
MODEL_NAME = "Qwen/Qwen2.5-3B"  # Uses <|im_start|> format  
MODEL_NAME = "meta-llama/Llama-3.1-8B"  # Uses [INST]/[/INST] format
```

## Dataset

Combined multiple Kotlin datasets from HuggingFace:
- **JetBrains/Kotlin_QA** - 47 Q&A samples
- **JetBrains/Kotlin_HumanEval** - 161 coding challenges
- **mvasiliniuc/iva-kotlin-codeint-clean** - ~4k code samples

- **Total**: 4,245 samples

## Hardware Requirements

| Model | VRAM Needed |
|-------|-------------|
| Qwen2.5-3B | ~12GB |
| Phi-2 (2.7B) | ~8GB |
| TinyLlama (1.1B) | ~4GB |

Tested on: NVIDIA RTX 5080 (16GB)

## Results

| Model | Base PPL | Fine-tuned PPL | Improvement |
|-------|----------|----------------|-------------|
| **Qwen2.5-3B** | 7.13 | 2.17 | **69.6%** |
| Phi-2 (2.7B) | 8.32 | 5.66 | 32% |

Training loss: 0.89

## GitHub

Repository: https://github.com/Skrilltrax/KotlinSage

## Next Steps

To improve the model further:

1. **More Data**: Add more Kotlin Q&A samples
2. **More Epochs**: Try 5-10 epochs
3. **Hyperparameter Tuning**: Adjust learning rate, LoRA rank
4. **Deploy**: Build an Android app using this model

## Acknowledgments

- [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) - Base model
- [JetBrains/Kotlin_QA](https://huggingface.co/datasets/JetBrains/Kotlin_QA) - Dataset
- [PEFT](https://github.com/huggingface/peft) - LoRA implementation