#!/usr/bin/env python3
"""
Evaluation script - measures model improvement via perplexity.
Automatically detects the base model and applies the correct format.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from math import exp
import json


def detect_base_model(model_path):
    """Detect base model from adapter config."""
    adapter_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_path):
        with open(adapter_path) as f:
            config = json.load(f)
            return config.get("base_model_name_or_path", "Qwen/Qwen2.5-3B")
    return "Qwen/Qwen2.5-3B"  # Default


def get_perplexity(model, tokenizer, dataset, config):
    """Calculate perplexity using model-specific format."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    prompt_format = config["prompt_format"]

    with torch.no_grad():
        for i, example in enumerate(dataset):
            q = example.get('Question') or example.get('question', '')
            a = example.get('Answer') or example.get('answer', '')
            text = prompt_format(q, a)

            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            encodings = {k: v.to(model.device) for k, v in encodings.items()}
            encodings["labels"] = encodings["input_ids"].clone()

            outputs = model(**encodings)

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = encodings["labels"][..., 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='sum'
            )

            valid_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item()
            total_tokens += valid_tokens

            if i % 10 == 0:
                torch.cuda.empty_cache()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return exp(avg_loss)


def main():
    from utils import get_model_config

    print("Loading dataset...")
    dataset = load_dataset("json", data_files="data/kotlin_combined.jsonl")["train"]

    print("\n" + "="*60)
    print("EVALUATION: Comparing base model vs fine-tuned model")
    print("="*60)

    # Detect base model
    base_model_path = detect_base_model("output/final")
    print(f"\nDetected base model: {base_model_path}")

    config = get_model_config(base_model_path)

    # Evaluate base model
    print("\n[1/2] Evaluating base model...")
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

    base_perplexity = get_perplexity(base_model, tokenizer, dataset, config)
    print(f"Base model perplexity: {base_perplexity:.2f}")

    del base_model
    torch.cuda.empty_cache()

    # Evaluate fine-tuned model
    print("\n[2/2] Evaluating fine-tuned model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        **config.get("tokenizer_config", {})
    )
    ft_model = PeftModel.from_pretrained(base_model, "output/final")

    ft_perplexity = get_perplexity(ft_model, tokenizer, dataset, config)
    print(f"Fine-tuned model perplexity: {ft_perplexity:.2f}")

    # Results
    improvement = (base_perplexity - ft_perplexity) / base_perplexity * 100

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Base model perplexity:      {base_perplexity:.2f}")
    print(f"Fine-tuned perplexity:      {ft_perplexity:.2f}")
    print(f"Improvement:                {improvement:.1f}%")

    if improvement > 0:
        print("\n✓ Model improved after fine-tuning!")
    else:
        print("\n✗ Model didn't improve")


if __name__ == "__main__":
    main()