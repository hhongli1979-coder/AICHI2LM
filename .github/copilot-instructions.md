# Copilot Instructions for TeleChat (星辰语义大模型)

This repository contains **TeleChat**, a Chinese/English bilingual large language model developed by China Telecom AI. Available in 7B and 12B versions with various quantization options.

## Tech Stack

- **Language**: Python
- **Framework**: PyTorch, Transformers (HuggingFace)
- **Training**: DeepSpeed (ZeRO optimization)
- **Acceleration**: Flash Attention 2
- **Quantization**: AutoGPTQ (Int4/Int8)
- **Deployment**: FastAPI, Uvicorn, Streamlit

## Project Structure

```
├── models/              # Model weights (7B, 12B, quantized versions)
├── deepspeed-telechat/  # DeepSpeed fine-tuning code
│   ├── sft/             # Supervised fine-tuning scripts
│   └── utils/           # Utility functions
├── inference_telechat/  # Inference demo scripts
├── service/             # API and Web deployment
│   ├── telechat_service.py  # FastAPI service
│   └── web_demo.py          # Streamlit web demo
├── evaluation/          # Model evaluation scripts (MMLU, C-Eval)
├── quant/               # Quantization code (GPTQ)
├── example_datas/       # Sample training data
└── docs/                # Documentation
```

## Setup and Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch==1.13.1`
- `transformers==4.30.0`
- `deepspeed==0.8.3`
- `flash-attn`
- `auto-gptq==0.3.0`

## Running Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

tokenizer = AutoTokenizer.from_pretrained('./models/12B', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    './models/12B',
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)
generate_config = GenerationConfig.from_pretrained('./models/12B')
answer, history = model.chat(
    tokenizer=tokenizer,
    question="Your question here",
    history=[],
    generation_config=generate_config,
    stream=False
)
```

## Running Services

```bash
# API Service
python service/telechat_service.py

# Web Demo
streamlit run service/web_demo.py
```

## Fine-tuning with DeepSpeed

1. Process data:
```bash
python deepspeed-telechat/sft/process_data.py \
    --data_path data.json \
    --tokenizer_path ./models/12B \
    --data_output_path $DATA_OUTPUT_PATH \
    --max_seq_len 4096 \
    --num_samples $NUM_SAMPLES
```

2. Run training:
```bash
deepspeed --master_port 29500 deepspeed-telechat/sft/main.py \
    --data_path $DATA_OUTPUT_PATH \
    --model_name_or_path ./models/12B \
    --per_device_train_batch_size 1 \
    --max_seq_len 4096 \
    --deepspeed \
    --output_dir $OUTPUT
```

## Coding Conventions

- Use **Python type hints** for function signatures
- Follow **PEP 8** style guidelines
- Add **docstrings** to functions and classes (Chinese or English)
- Use `trust_remote_code=True` when loading TeleChat models
- Handle CUDA/GPU availability gracefully with device checks
- Use `torch.float16` or `torch.bfloat16` for model inference to save memory

## Special Tokens

TeleChat uses these special tokens:
- `<_end>` - End of generation
- `<_user>` - User question marker
- `<_bot>` - Model response marker

## Important Notes

- Always use `trust_remote_code=True` when loading models
- The tokenizer uses BBPE algorithm with a vocabulary size of 160,256
- Flash Attention 2 improves training speed by ~20%
- ZeRO-3 with gradient checkpointing is recommended for full fine-tuning
- Support for both single-turn and multi-turn conversations

## Do Not

- Do not commit model weights to the repository
- Do not hardcode API keys or credentials
- Do not remove the `trust_remote_code=True` parameter
- Do not use model for any activities that violate laws or harm national security
