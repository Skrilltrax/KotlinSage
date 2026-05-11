"""
Model-specific configurations for different base models.
Dynamically detects and applies the correct prompts, tokenizers, and settings.
"""

def get_model_config(model_name):
    """
    Get the correct configuration based on the model name.

    Args:
        model_name: Name of the base model (e.g., "microsoft/Phi-2", "Qwen/Qwen2.5-3B")

    Returns:
        Dictionary with:
        - prompt_format: Function to format training prompts
        - prompt_format_inference: Function to format inference prompts
        - eos_token: End-of-sequence token
        - use_fast: Whether to use fast tokenizer
    """
    model_name_lower = model_name.lower()

    if "qwen" in model_name_lower:
        return {
            "prompt_format": lambda q, a: f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>",
            "prompt_format_inference": lambda q: f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n",
            "extract_response": lambda r: r.split("<|im_start|>assistant\n")[1] if "<|im_start|>assistant\n" in r else r,
            "tokenizer_config": {"trust_remote_code": True},
            "pad_token": "<|endoftext|>",
        }
    elif "phi" in model_name_lower:
        return {
            "prompt_format": lambda q, a: f"<|user|>\n{q}\n<|answer|>\n{a}<|endoftext|>",
            "prompt_format_inference": lambda q: f"<|user|>\n{q}\n<|answer|>",
            "extract_response": lambda r: r.split("<|answer|>")[1] if "<|answer|>" in r else r,
            "tokenizer_config": {"trust_remote_code": True},
            "pad_token": None,  # Use eos_token
        }
    elif "llama" in model_name_lower or "mistral" in model_name_lower:
        return {
            "prompt_format": lambda q, a: f"[INST] {q} [/INST] {a}",
            "prompt_format_inference": lambda q: f"[INST] {q} [/INST]",
            "extract_response": lambda r: r.split("[/INST]")[-1].strip() if "[/INST]" in r else r,
            "tokenizer_config": {"trust_remote_code": True},
            "pad_token": None,
        }
    else:
        # Default fallback - try generic chat format
        return {
            "prompt_format": lambda q, a: f"Question: {q}\nAnswer: {a}",
            "prompt_format_inference": lambda q: f"Question: {q}\nAnswer:",
            "extract_response": lambda r: r.split("Answer:")[-1].strip() if "Answer:" in r else r,
            "tokenizer_config": {},
            "pad_token": None,
        }