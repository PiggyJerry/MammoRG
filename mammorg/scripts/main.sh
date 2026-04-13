#!/bin/bash

export MODEL_NAME="mammorg"
export BASE_DIR="/home/user/MammoRG-main"
export PYTHONPATH=".:$PYTHONPATH"
STAGE1_SCRIPT="./scripts/stage1.sh"
STAGE2_ALIGNMENT_SCRIPT="./scripts/stage2_alignment.sh"
STAGE2_FINETUNE_SCRIPT="./scripts/stage2_finetune.sh"
EVAL_SCRIPT="./scripts/eval.sh"

check_script() {
    if [ ! -f "$1" ]; then
        echo "Error: no script: $1"
        exit 1
    fi
}
check_script "$STAGE1_SCRIPT"
check_script "$STAGE2_ALIGNMENT_SCRIPT"
check_script "$STAGE2_FINETUNE_SCRIPT"
check_script "$EVAL_SCRIPT"

prepare_script() {
    if [ ! -f "$1" ]; then  
        echo "Error: no script: $1"
        exit 1
    fi
    chmod +x "$1"
}

prepare_script "$STAGE1_SCRIPT"
prepare_script "$STAGE2_ALIGNMENT_SCRIPT"
prepare_script "$STAGE2_FINETUNE_SCRIPT"
prepare_script "$EVAL_SCRIPT"

echo "================================"
echo "Cross-Modal Retrieval Training Stage (Stage 1)"
echo "================================"
if ! "$STAGE1_SCRIPT"; then
    echo "Failed"
    exit 1
fi

echo "================================"
echo "Supervised Fine-Tuning Stage (Stage 2 Alignment)"
echo "================================"
if ! "$STAGE2_ALIGNMENT_SCRIPT"; then
    echo "Failed"
    exit 1
fi

echo "================================"
echo "Supervised Fine-Tuning Stage (Stage 2 Fine-Tune)"
echo "================================"
if ! "$STAGE2_FINETUNE_SCRIPT"; then
    echo "Failed"
    exit 1
fi

echo "================================"
echo "Evaluation"
echo "================================"
if ! "$EVAL_SCRIPT"; then
    echo "Failed"
fi

echo "================================"
echo "All done!"
echo "================================"
