#!/usr/bin/env python3
"""
KotlinSage Fine-tuning Script
==============================
This script fine-tunes Phi-2 on Kotlin Q&A data using LoRA (Low-Rank Adaptation).

What is LoRA?
-------------
LoRA is a parameter-efficient fine-tuning technique. Instead of updating all model
weights (which would require huge GPU memory), we add small "adapter" matrices
that capture task-specific knowledge. The original weights stay frozen.

How it works:
- Original model: frozen, unchanged
- LoRA adapters: small matrices (rank r=16) that get trained
- At inference, we merge adapters into the model for seamless use

This reduces memory usage from ~20GB to ~6GB for a 2.7B model!
"""

import torch
from transformers import (
    AutoModelForCausalLM,    # Loads causal language models (GPT-style)
    AutoTokenizer,          # Tokenizer for text <-> token conversion
    TrainingArguments,      # Hyperparameters for training
    DataCollatorForLanguageModeling  # Batches samples for training
)
from peft import (
    LoraConfig,             # LoRA configuration
    get_peft_model,         # Wraps model with LoRA adapters
    TaskType                # Type of task (CAUSAL_LM = next token prediction)
)
from datasets import load_dataset
import sys
import os

# Add parent directory to path for utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_model_config

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-3B"       # Base model to fine-tune
OUTPUT_DIR = "./output"            # Where to save checkpoints
MAX_LEN = 512                      # Max tokens per sample (truncation)
LORA_R = 16                        # LoRA rank - higher = more capacity, more memory
LORA_ALPHA = 16                    # LoRA scaling factor
LORA_DROPOUT = 0.1                 # Dropout for LoRA layers


# ============================================================================
# PROMPT FORMATTING
# ============================================================================

def format_prompt(example):
    """
    Format Q&A pairs into a chat template that Phi-2 understands.

    Phi-2 uses <|user|> and <|assistant|> tokens to distinguish between
    user input and model output. This helps the model learn when to respond.

    Example:
    <|user|>
    How do I create a singleton in Kotlin?
    <|answer|>
    Use object declaration:
    object MySingleton {
        fun hello() = "Hello!"
    }
    <|endoftext|>
    """
    return f"""<|user|>
{example['Question']}
<|answer|>
{example['Answer']}<|endoftext|>"""


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    # -------------------------------------------------------------------------
    # Step 1: Load Tokenizer
    # -------------------------------------------------------------------------
    # Tokenizer converts text to token IDs that the model can process
    # It also handles special tokens like <|endoftext|>
    # Get model-specific configuration
    config = get_model_config(MODEL_NAME)
    print(f"Using model config for: {MODEL_NAME}")

    # Load tokenizer with model-specific settings
    tokenizer_config = config.get("tokenizer_config", {})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **tokenizer_config)

    # Set pad token
    if config.get("pad_token"):
        tokenizer.pad_token = config["pad_token"]
    else:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # Step 2: Load Model
    # -------------------------------------------------------------------------
    # AutoModelForCausalLM loads a transformer that predicts next token
    # device_map="auto" automatically distributes model across GPU/CPU
    # torch.float16 uses half-precision (less memory, faster)
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,  # FP16 = 2 bytes per param vs 4 bytes for FP32
        trust_remote_code=True      # Required for some models
    )

    # -------------------------------------------------------------------------
    # Step 3: Load and Tokenize Dataset
    # -------------------------------------------------------------------------
    # Load the Kotlin Q&A JSONL file we created earlier
    print("Loading dataset...")
    dataset = load_dataset("json", data_files="data/kotlin_combined.jsonl")["train"]

    def tokenize(examples):
        """
        Convert text to token IDs.

        - truncation=True: Cut off text longer than MAX_LEN
        - padding="max_length": Pad all samples to same length for batching
        """
        # Handle both lowercase and uppercase keys (new datasets use lowercase)
        q = examples.get('Question') or examples.get('question', '')
        a = examples.get('Answer') or examples.get('answer', '')

        # Use model-specific prompt format
        prompt = config["prompt_format"](q, a)

        return tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length"
        )

    # Apply tokenization to all examples (batched=False for simplicity)
    # remove_columns removes original text columns - we only keep token IDs
    dataset = dataset.map(tokenize, batched=False, remove_columns=dataset.column_names)

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Token length: {dataset[0]['input_ids'].__len__()} tokens")

    # -------------------------------------------------------------------------
    # Step 4: Configure LoRA
    # -------------------------------------------------------------------------
    # LoRA works by adding small matrices to attention layers:
    # - q_proj: Query projection (what we're looking for)
    # - k_proj: Key projection (what we're comparing against)
    # - v_proj: Value projection (what information to pass)
    # - o_proj: Output projection (final attention output)
    #
    # Instead of W (full matrix), we learn W = A @ B where:
    # - A is (d x r) - reduces from d to low rank r
    # - B is (r x d) - expands back to d
    # Total params: 2 * d * r instead of d * d
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,      # Next-token prediction
        r=LORA_R,                           # Rank (16 is good default)
        lora_alpha=LORA_ALPHA,              # Scaling factor
        lora_dropout=LORA_DROPOUT,          # Regularization
        target_modules=[                   # Which layers to add adapters to
            "q_proj",   # Query in attention
            "k_proj",   # Key in attention
            "v_proj",   # Value in attention
            "o_proj"    # Output projection
        ],
        bias="none"  # Don't train bias terms
    )

    # Wrap original model with LoRA adapters
    # This freezes most weights and only trains the adapters
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing to save memory
    model.base_model._set_gradient_checkpointing(True)

    # Show how many parameters we're actually training
    # Should be ~1-2% of total model params
    model.print_trainable_parameters()

    # -------------------------------------------------------------------------
    # Step 5: Configure Training Arguments
    # -------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,              # Where to save checkpoints
        per_device_train_batch_size=1,       # Reduce to save memory
        gradient_accumulation_steps=4,       # Effective batch = 1 * 4 = 4 samples
        num_train_epochs=3,                 # How many times to see all data
        learning_rate=3e-4,                  # How fast to update weights
        # Higher LR for LoRA (3e-4 typical), lower for full fine-tune (1e-5)
        logging_steps=10,                   # Log every 10 steps
        save_steps=100,                      # Save checkpoint every 100 steps
        save_total_limit=2,                  # Keep only 2 checkpoints
        fp16=True,                           # Use half-precision training
        warmup_steps=10,                    # Gradual learning rate increase
        report_to="none",                   # Disable wandb/tensorboard
        gradient_checkpointing=True          # Save memory by recomputing
    )

    # DataCollator creates batches from the dataset
    # mlm=False means we're doing causal LM (predict next token),
    # not masked LM (fill in the blank)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # -------------------------------------------------------------------------
    # Step 6: Create Trainer and Train
    # -------------------------------------------------------------------------
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    print("Starting training...")
    print(f"Total steps: {len(dataset) // 16 * 3}")  # (samples / batch) * epochs

    # This is where the magic happens!
    # The model learns to answer Kotlin questions
    trainer.train()

    # -------------------------------------------------------------------------
    # Step 7: Save the Fine-tuned Model
    # -------------------------------------------------------------------------
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"Done! Model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()