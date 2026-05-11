#!/usr/bin/env python3
"""
Inference script for KotlinSage fine-tuned model.

This script loads the fine-tuned model and generates responses to Kotlin questions.
Supports multiple base models (Phi-2, Qwen, Llama, etc.) automatically.

Usage:
    python scripts/inference.py [model_path]

Example:
    python scripts/inference.py output/final
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import get_model_config

DEFAULT_MODEL_PATH = "output/final"

def load_model(model_path=DEFAULT_MODEL_PATH):
    """Load the fine-tuned model with LoRA adapters."""
    # Get base model from the adapter config
    base_model_path = None

    # Try to detect base model from adapter config
    adapter_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_path):
        import json
        with open(adapter_path) as f:
            config = json.load(f)
            base_model_path = config.get("base_model_name_or_path", "")

    if not base_model_path:
        # Default to Qwen2.5-3B if not found
        base_model_path = "Qwen/Qwen2.5-3B"

    print(f"Detected base model: {base_model_path}")

    # Get model configuration
    config = get_model_config(base_model_path)

    print(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        **config.get("tokenizer_config", {})
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, **config.get("tokenizer_config", {}))
    if config.get("pad_token"):
        tokenizer.pad_token = config["pad_token"]
    else:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading fine-tuned adapters from {model_path}...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    return model, tokenizer, config


def generate(prompt, model, tokenizer, config, max_new_tokens=256):
    """Generate a response using model-specific format."""
    formatted = config["prompt_format_inference"](prompt)

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return config["extract_response"](response)


def test_model(model, tokenizer, config):
    """Run test questions through the model."""
    test_questions = [
        "How do I create a singleton in Kotlin?",
        "What is the difference between val and var?",
        "How do I use coroutines in Kotlin?",
    ]

    print("\n" + "="*60)
    print("TESTING FINE-TUNED MODEL")
    print("="*60)

    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question}")
        print(f"A: {generate(question, model, tokenizer, config)}")
        print("-"*40)


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    model, tokenizer, config = load_model(model_path)
    test_model(model, tokenizer, config)


if __name__ == "__main__":
    main()