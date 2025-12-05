# TeleChat 快速开始

本文档提供 TeleChat 最常用命令的快速参考。详细说明请查看 [完整命令参考](./COMMANDS.md)。

## 🚀 一键部署

```bash
# 最简单的方式 - 一键启动 API 和 Web 服务
python deploy.py

# 访问服务
# API 文档: http://localhost:8070/docs
# Web 界面: http://localhost:8501
```

## 📥 安装依赖

```bash
pip install -r requirements.txt
```

## 💬 快速推理

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
PATH = 'models/7B'  # 或 models/12B

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    PATH, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
)
generate_config = GenerationConfig.from_pretrained(PATH)

# 对话
question = "你好，请介绍一下你自己"
answer, history = model.chat(
    tokenizer=tokenizer, question=question, history=[],
    generation_config=generate_config, stream=False
)
print(answer)
```

## 🎯 训练模型

```bash
cd deepspeed-telechat/sft

# 1. 处理数据
python process_data.py \
    --data_path data.json \
    --tokenizer_path ../../models/12B \
    --data_output_path datas/data_files \
    --max_seq_len 4096 \
    --num_samples 10000 \
    --num_workers 10 \
    --process_method multiple

# 2. 单机多卡训练
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash run_telechat_single_node.sh
```

## ⚡ 模型量化

```python
from transformers import AutoTokenizer
from auto_gptq import BaseQuantizeConfig
from modeling_telechat_gptq import TelechatGPTQForCausalLM

# Int4 量化
tokenizer_path = 'models/7B'
pretrained_model_dir = 'models/7B'
quantized_model_dir = 'models/7B-int4'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
examples = [tokenizer("auto-gptq is an easy-to-use model quantization library.")]

quantize_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)
model = TelechatGPTQForCausalLM.from_pretrained(
    pretrained_model_dir, quantize_config, trust_remote_code=True
)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
```

## 🔧 常见配置

### 指定 GPU

```bash
# 使用单个 GPU
export CUDA_VISIBLE_DEVICES=0
python deploy.py --gpu 0

# 使用多个 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python deploy.py --gpu 0,1,2,3
```

### 指定模型路径

```bash
python deploy.py --model-path models/12B
```

### 自定义端口

```bash
python deploy.py --api-port 8080 --web-port 8502
```

## 📊 模型评测

```bash
cd evaluation

# C-Eval
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
unzip ceval-exam.zip
python score_CEVAL.py --path ../models/7B --five_shot

# MMLU
python score_MMLU.py
```

## 🌐 服务部署

### 启动 API 服务

```bash
cd service
python telechat_service.py
# API 文档: http://localhost:8070/docs
```

### 启动 Web 界面

```bash
cd service
streamlit run web_demo.py
# Web 界面: http://localhost:8501
```

## 📖 更多信息

- **完整命令参考**: [COMMANDS.md](./COMMANDS.md)
- **详细教程**: [docs/tutorial.md](./docs/tutorial.md)
- **部署指南**: [DEPLOYMENT.md](./DEPLOYMENT.md)
- **模型下载**: https://huggingface.co/Tele-AI

## 🆘 常见问题

### GPU 内存不足？

使用量化模型：
```bash
python deploy.py --model-path models/7B-int4
```

### 端口被占用？

使用其他端口：
```bash
python deploy.py --api-port 8080 --web-port 8502
```

### 模型加载失败？

检查模型文件是否完整：
```bash
ls -lh models/7B/
# 应该包含: config.json, pytorch_model.bin, tokenizer.model 等
```

---

**提示**: 所有命令的详细说明和高级用法请参考 [COMMANDS.md](./COMMANDS.md)
