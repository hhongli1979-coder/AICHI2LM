#!/bin/bash
# TeleChat 评估脚本

set -e

# 默认配置
MODEL_PATH=${MODEL_PATH:-"./models/12B"}
EVAL_TYPE=${1:-"all"}

echo "=========================================="
echo "TeleChat Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Evaluation Type: $EVAL_TYPE"
echo "=========================================="

cd evaluation

case $EVAL_TYPE in
    "mmlu")
        echo "Running MMLU evaluation..."
        python3 score_MMLU.py --model_path $MODEL_PATH
        ;;
    "ceval")
        echo "Running C-Eval evaluation..."
        python3 score_CEVAL.py --model_path $MODEL_PATH
        ;;
    "all")
        echo "Running all evaluations..."
        python3 score_MMLU.py --model_path $MODEL_PATH
        python3 score_CEVAL.py --model_path $MODEL_PATH
        ;;
    *)
        echo "Unknown evaluation type: $EVAL_TYPE"
        echo "Available options: mmlu, ceval, all"
        exit 1
        ;;
esac

echo "Evaluation completed!"
