#!/bin/bash
#
# KotlinSage CLI
# Usage: ./cli.sh [setup|train|eval|infer|all]
#

set -e

# Get project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source environment
source .venv/bin/activate
source ~/dotfiles/exports/exports.zsh 2>/dev/null || true

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

cmd_setup() {
    log_info "Setting up KotlinSage environment..."

    # Create venv if needed
    if [ ! -d ".venv" ]; then
        log_info "Creating virtual environment..."
        uv venv
    fi

    log_info "Installing dependencies..."
    uv add torch transformers peft bitsandbytes accelerate datasets

    log_info "Downloading dataset..."
    python scripts/download_dataset.py

    echo ""
    log_info "✓ Setup complete!"
}

cmd_train() {
    log_info "Training model (10 epochs)..."
    python scripts/train.py
    echo ""
    log_info "✓ Training complete! Model saved to output/final"
}

cmd_eval() {
    log_info "Evaluating model perplexity..."
    python scripts/evaluate.py
}

cmd_infer() {
    log_info "Testing inference..."
    python scripts/inference.py output/final
}

cmd_all() {
    log_info "Running full pipeline..."
    cmd_setup
    cmd_train
    cmd_eval
    cmd_infer
    echo ""
    echo "=================================================="
    log_info "All done! KotlinSage is ready."
    echo "=================================================="
}

# Main
case "${1:-}" in
    setup)    cmd_setup ;;
    train)    cmd_train ;;
    eval|evaluate)  cmd_eval ;;
    infer|inference)  cmd_infer ;;
    all)      cmd_all ;;
    *)
        echo "KotlinSage CLI"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  setup    - Setup environment and download data"
        echo "  train    - Train the model"
        echo "  eval     - Evaluate perplexity"
        echo "  infer    - Test inference"
        echo "  all      - Run full pipeline"
        echo ""
        echo "Example:"
        echo "  $0 all   # Runs setup, train, eval, infer"
        exit 1
        ;;
esac