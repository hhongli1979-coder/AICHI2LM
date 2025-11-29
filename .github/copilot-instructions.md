# Copilot Instructions for TeleChat Repository

This repository contains the TeleChat (星辰语义大模型) large language model developed by China Telecom AI Technology Co., Ltd. It includes model inference, fine-tuning, quantization, and deployment code.

## Project Overview

- **Language**: Primarily Python
- **Framework**: PyTorch, Transformers (Hugging Face), DeepSpeed
- **Purpose**: Training, inference, and deployment of TeleChat 7B and 12B language models

## Code Style and Conventions

- Follow PEP 8 style guidelines for Python code
- Use meaningful variable and function names in English
- Keep comments and documentation in Chinese where they already exist; new documentation can be in English or Chinese
- Preserve existing code patterns and structure

## Key Directories

- `deepspeed-telechat/`: DeepSpeed-based fine-tuning code
- `inference_telechat/`: Model inference code
- `models/`: Model architecture definitions
- `quant/`: Quantization code (GPTQ-based)
- `service/`: API and web service deployment
- `evaluation/`: Evaluation scripts
- `docs/`: Documentation

## Dependencies

- PyTorch
- Transformers
- DeepSpeed
- auto-gptq (for quantization)
- FlashAttention2

## Important Considerations

- This codebase supports both single-GPU and multi-GPU inference
- Training supports distributed training with DeepSpeed Zero optimization
- Model supports int4 and int8 quantization via AutoGPTQ
- The tokenizer uses BBPE algorithm with special tokens: `<_end>`, `<_user>`, `<_bot>`

## Testing

When making changes, ensure:
- Code runs without errors
- Existing functionality is preserved
- New features follow existing patterns in the codebase
