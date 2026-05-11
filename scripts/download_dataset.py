#!/usr/bin/env python3
"""
Download and combine multiple Kotlin datasets for fine-tuning.

This script loads multiple Kotlin datasets from HuggingFace and combines them
into a single JSONL file for training.

Datasets:
1. JetBrains/Kotlin_QA - Q&A pairs (47 samples)
2. JetBrains/Kotlin_HumanEval - Coding challenges (161 samples)
3. mvasiliniuc/iva-kotlin-codeint-clean - Code samples (202k samples, we use a subset)

For the large dataset, we sample to keep training manageable.
"""
import json
import os
from datasets import load_dataset

def load_kotlin_qa():
    """Load Kotlin Q&A dataset from JetBrains."""
    print("Loading JetBrains/Kotlin_QA...")
    dataset = load_dataset("JetBrains/Kotlin_QA")
    return [{"question": ex["Question"], "answer": ex["Answer"]} for ex in dataset["train"]]

def load_kotlin_humaneval():
    """Load Kotlin HumanEval dataset (coding challenges)."""
    print("Loading JetBrains/Kotlin_HumanEval...")
    dataset = load_dataset("JetBrains/Kotlin_HumanEval")

    # Convert to Q&A format
    # HumanEval has prompt (task description) and canonical_solution
    results = []
    for ex in dataset["train"]:
        # Create a question about solving the coding task
        question = f"Solve this Kotlin coding problem: {ex['prompt']}"
        answer = ex.get("canonical_solution", ex.get("test", ""))
        results.append({"question": question, "answer": answer})

    return results

def load_kotlin_code():
    """Load Kotlin code dataset and convert to Q&A format."""
    print("Loading mvasiliniuc/iva-kotlin-codeint-clean...")

    # For code datasets, we create "explain this code" style Q&A
    # This helps model learn to explain Kotlin code
    dataset = load_dataset("mvasiliniuc/iva-kotlin-codeint-clean", split="train")

    # Sample a subset (keep training manageable)
    # Take every Nth sample to get diverse examples
    sampled = dataset.select(range(0, len(dataset), 50))  # ~4k samples

    results = []
    for ex in sampled:
        # The dataset has code in various fields, try to find code content
        # Common field names: code, content, input, etc.
        code = ex.get("content") or ex.get("code") or str(ex)

        # Create a prompt to explain/describe the code
        question = "Explain this Kotlin code:"
        # Truncate long code samples
        code_preview = code[:500] if len(code) > 500 else code
        answer = f"Here is the Kotlin code:\n```{code_preview}```"

        results.append({"question": question, "answer": answer})

    return results

def combine_datasets():
    """Combine all datasets into one JSONL file."""
    all_data = []

    # Load each dataset
    try:
        qa_data = load_kotlin_qa()
        print(f"  -> Got {len(qa_data)} Q&A samples")
        all_data.extend(qa_data)
    except Exception as e:
        print(f"  -> Failed to load Kotlin_QA: {e}")

    try:
        humaneval_data = load_kotlin_humaneval()
        print(f"  -> Got {len(humaneval_data)} HumanEval samples")
        all_data.extend(humaneval_data)
    except Exception as e:
        print(f"  -> Failed to load Kotlin_HumanEval: {e}")

    try:
        code_data = load_kotlin_code()
        print(f"  -> Got {len(code_data)} code samples")
        all_data.extend(code_data)
    except Exception as e:
        print(f"  -> Failed to load kotlin-codeint: {e}")

    print(f"\nTotal combined: {len(all_data)} samples")

    # Create output directory
    os.makedirs("data", exist_ok=True)

    # Save to JSONL
    output_path = "data/kotlin_combined.jsonl"
    with open(output_path, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved to {output_path}")
    return len(all_data)

def main():
    print("="*60)
    print("Downloading and combining Kotlin datasets")
    print("="*60)
    total = combine_datasets()
    print(f"\n✓ Done! Combined {total} samples for training")

if __name__ == "__main__":
    main()