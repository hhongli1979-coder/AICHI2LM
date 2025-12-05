# TeleChat 常用命令参考

本文档提供了 TeleChat 模型的所有常用命令和操作指南。

## 目录

- [环境安装](#环境安装)
- [模型下载](#模型下载)
- [快速部署](#快速部署)
- [模型推理](#模型推理)
- [模型训练](#模型训练)
- [模型量化](#模型量化)
- [模型评测](#模型评测)
- [数据处理](#数据处理)
- [服务部署](#服务部署)

---

## 环境安装

### 安装依赖包

```bash
# 安装所有依赖
pip install -r requirements.txt

# 或者手动安装主要依赖
pip install torch transformers deepspeed auto-gptq
```

### 验证环境

```bash
# 检查 Python 版本（需要 3.8+）
python --version

# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 检查 GPU 设备
nvidia-smi
```

---

## 模型下载

### 从 HuggingFace 下载

```bash
# 使用 git-lfs 下载模型
git lfs install
git clone https://huggingface.co/Tele-AI/Telechat-7B models/7B

# 下载其他版本
git clone https://huggingface.co/Tele-AI/TeleChat-12B models/12B
git clone https://huggingface.co/Tele-AI/Telechat-7B-int8 models/7B-int8
git clone https://huggingface.co/Tele-AI/Telechat-7B-int4 models/7B-int4
```

### 使用 huggingface-cli 下载

```bash
# 安装 huggingface-cli
pip install huggingface_hub

# 下载模型
huggingface-cli download Tele-AI/Telechat-7B --local-dir models/7B
```

---

## 快速部署

### 一键部署（推荐）

```bash
# 使用 Python 脚本（跨平台）
python deploy.py

# 使用配置文件
python deploy.py --config deploy_config.yaml

# 指定模型路径
python deploy.py --model-path models/12B

# 指定 GPU 设备
python deploy.py --gpu 0,1

# 指定服务端口
python deploy.py --api-port 8080 --web-port 8502

# 查看所有选项
python deploy.py --help
```

### Shell 脚本部署（Linux/Mac）

```bash
# 添加执行权限
chmod +x deploy.sh

# 使用默认配置
./deploy.sh

# 指定模型和 GPU
./deploy.sh --model models/12B --gpu 0,1

# 指定端口
./deploy.sh --api-port 8080 --web-port 8502

# 查看帮助
./deploy.sh --help
```

### Windows 批处理部署

```cmd
REM 直接运行
deploy.bat

REM 设置环境变量后运行
set MODEL_PATH=models\12B
set CUDA_VISIBLE_DEVICES=0
deploy.bat
```

### 访问服务

```bash
# API 文档地址
http://localhost:8070/docs

# Web 界面地址
http://localhost:8501

# 停止服务：按 Ctrl+C
```

---

## 模型推理

### Python 交互式推理

```python
# 基础推理示例
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# 设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 加载模型
PATH = 'models/7B'  # 或 models/12B
tokenizer = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    PATH, 
    trust_remote_code=True, 
    device_map="auto",
    torch_dtype=torch.float16
)
generate_config = GenerationConfig.from_pretrained(PATH)
model.eval()

# 单轮对话
question = "你是谁？"
answer, history = model.chat(
    tokenizer=tokenizer, 
    question=question, 
    history=[], 
    generation_config=generate_config,
    stream=False
)
print(f"回答: {answer}")

# 多轮对话
question = "你有什么功能？"
answer, history = model.chat(
    tokenizer=tokenizer, 
    question=question, 
    history=history,  # 传入历史对话
    generation_config=generate_config,
    stream=False
)
print(f"回答: {answer}")

# 流式输出
question = "介绍一下深度学习"
for response, history in model.chat(
    tokenizer=tokenizer,
    question=question,
    history=[],
    generation_config=generate_config,
    stream=True
):
    print(response, end='', flush=True)
```

### 命令行推理

```bash
# 运行推理脚本
cd inference_telechat
python telechat_infer_demo.py

# 指定模型路径（需修改脚本中的 PATH 变量）
# 编辑 telechat_infer_demo.py，修改：PATH = '../models/12B'
```

### 多 GPU 推理

```python
# 使用多个 GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

# device_map="auto" 会自动分配模型到多个 GPU
model = AutoModelForCausalLM.from_pretrained(
    PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)
```

---

## 模型训练

### 数据准备

```bash
# 准备训练数据
# 1. 单轮数据格式（example_datas/single_turn_example.jsonl）：
# {"text": "用户问题\n模型回答"}

# 2. 多轮数据格式（example_datas/multi_turn_example.jsonl）：
# {"text": "<_user>问题1<_bot>回答1<_user>问题2<_bot>回答2"}

# 配置数据权重（data.json）
cat > deepspeed-telechat/sft/data.json << EOF
{
  "example_datas/single_turn_example.jsonl": 2.0,
  "example_datas/multi_turn_example.jsonl": 1.0
}
EOF
```

### 数据处理

```bash
cd deepspeed-telechat/sft

# 单进程处理数据
python process_data.py \
    --data_path data.json \
    --tokenizer_path ../../models/12B \
    --data_output_path datas/data_files \
    --max_seq_len 4096 \
    --num_samples 1000 \
    --num_workers 1 \
    --process_method single \
    --seed 42

# 多进程处理数据（更快）
python process_data.py \
    --data_path data.json \
    --tokenizer_path ../../models/12B \
    --data_output_path datas/data_files \
    --max_seq_len 4096 \
    --num_samples 10000 \
    --num_workers 10 \
    --process_method multiple \
    --seed 42
```

### 单机单卡训练

```bash
cd deepspeed-telechat/sft

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 训练
deepspeed --master_port 29500 main.py \
    --data_path datas/data_files \
    --model_name_or_path ../../models/7B \
    --with_loss_mask \
    --per_device_train_batch_size 1 \
    --max_seq_len 2048 \
    --learning_rate 3e-5 \
    --weight_decay 0.0001 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --precision fp16 \
    --warmup_proportion 0.1 \
    --gradient_checkpointing \
    --seed 1233 \
    --zero_stage 2 \
    --save_steps 100 \
    --deepspeed \
    --output_dir output
```

### 单机多卡训练

```bash
cd deepspeed-telechat/sft

# 使用脚本训练（推荐）
# 先编辑 run_telechat_single_node.sh 配置参数
bash run_telechat_single_node.sh

# 或手动指定 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed --master_port 29500 main.py \
    --data_path datas/data_files \
    --model_name_or_path ../../models/12B \
    --with_loss_mask \
    --per_device_train_batch_size 1 \
    --max_seq_len 4096 \
    --learning_rate 3e-5 \
    --weight_decay 0.0001 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --precision fp16 \
    --warmup_proportion 0.1 \
    --gradient_checkpointing \
    --offload \
    --seed 1233 \
    --zero_stage 3 \
    --save_steps 10 \
    --deepspeed \
    --output_dir output \
    2>&1 | tee output/training.log
```

### 多机多卡训练

```bash
cd deepspeed-telechat/sft

# 1. 准备 hostfile（my_hostfile）
cat > my_hostfile << EOF
worker0 slots=8
worker1 slots=8
EOF

# 2. 运行训练
deepspeed --master_port 29500 --hostfile=my_hostfile main.py \
    --data_path datas/data_files \
    --model_name_or_path ../../models/12B \
    --with_loss_mask \
    --per_device_train_batch_size 1 \
    --max_seq_len 4096 \
    --learning_rate 3e-5 \
    --weight_decay 0.0001 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --precision fp16 \
    --warmup_proportion 0.1 \
    --gradient_checkpointing \
    --offload \
    --seed 1233 \
    --zero_stage 3 \
    --save_steps 10 \
    --deepspeed \
    --output_dir output
```

### LoRA 微调

```bash
cd deepspeed-telechat/sft

# 使用 LoRA 训练脚本
bash run_telechat_lora.sh
```

### 监控训练进度

```bash
# 查看训练日志
tail -f output/training.log

# 使用 tensorboard（如果有）
tensorboard --logdir output

# 查看 GPU 使用情况
watch -n 1 nvidia-smi
```

---

## 模型量化

### Int8 量化

```python
# 离线量化
from transformers import AutoTokenizer
from auto_gptq import BaseQuantizeConfig
from modeling_telechat_gptq import TelechatGPTQForCausalLM

# 配置路径
tokenizer_path = 'models/7B'
pretrained_model_dir = 'models/7B'
quantized_model_dir = 'models/7B-int8'

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# 准备校准数据
calibration_text = [
    "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
]
examples = [tokenizer(_) for _ in calibration_text]

# 配置量化参数
quantize_config = BaseQuantizeConfig(
    bits=8,  # 8-bit 量化
    group_size=128,
    desc_act=False
)

# 加载模型并量化
model = TelechatGPTQForCausalLM.from_pretrained(
    pretrained_model_dir, 
    quantize_config,
    trust_remote_code=True
)
model.quantize(examples)

# 保存量化模型
model.save_quantized(quantized_model_dir)
print(f"模型已保存到: {quantized_model_dir}")
```

### Int4 量化

```python
# 离线量化
from transformers import AutoTokenizer
from auto_gptq import BaseQuantizeConfig
from modeling_telechat_gptq import TelechatGPTQForCausalLM

tokenizer_path = 'models/7B'
pretrained_model_dir = 'models/7B'
quantized_model_dir = 'models/7B-int4'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
calibration_text = [
    "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
]
examples = [tokenizer(_) for _ in calibration_text]

# 配置 4-bit 量化
quantize_config = BaseQuantizeConfig(
    bits=4,  # 4-bit 量化
    group_size=128,
    desc_act=False
)

model = TelechatGPTQForCausalLM.from_pretrained(
    pretrained_model_dir,
    quantize_config,
    trust_remote_code=True
)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
print(f"模型已保存到: {quantized_model_dir}")
```

### 量化模型推理

```python
# Int8/Int4 量化模型推理
from transformers import AutoTokenizer, GenerationConfig
from modeling_telechat_gptq import TelechatGPTQForCausalLM

PATH = 'models/7B-int8'  # 或 models/7B-int4

# 加载量化模型
tokenizer = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
model = TelechatGPTQForCausalLM.from_quantized(
    PATH,
    device="cuda:0",
    inject_fused_mlp=False,
    inject_fused_attention=False,
    trust_remote_code=True
)
generate_config = GenerationConfig.from_pretrained(PATH)
model.eval()

# 推理
question = "生抽与老抽的区别？"
answer, history = model.chat(
    tokenizer=tokenizer,
    question=question,
    history=[],
    generation_config=generate_config,
    stream=False
)
print("回答:", answer)
```

### 使用命令行量化

```bash
cd quant

# 编辑 quant.py 配置路径和量化参数
# 运行量化
python quant.py

# 测试量化模型
python telechat_quantized_infer_demo.py
```

---

## 模型评测

### C-Eval 评测

```bash
cd evaluation

# 1. 下载数据集
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
unzip ceval-exam.zip

# 2. 运行评测（5-shot）
python score_CEVAL.py --path ../models/7B --five_shot

# 3. 提交结果
# 将生成的 submission.json 提交到 https://cevalbenchmark.com/
```

### MMLU 评测

```bash
cd evaluation

# 1. 下载数据集
# 从 https://github.com/hendrycks/test 下载数据

# 2. 运行评测
python score_MMLU.py

# 参数说明（需修改脚本）：
# - model_path: 模型路径
# - data_dir: MMLU 数据集路径
```

### 自定义评测

```python
# 使用模型进行批量评测
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = 'models/7B'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# 批量评测问题
questions = [
    "什么是深度学习？",
    "Python 的主要特点是什么？",
    "如何提高模型性能？"
]

for question in questions:
    answer, _ = model.chat(
        tokenizer=tokenizer,
        question=question,
        history=[],
        stream=False
    )
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

---

## 数据处理

### 单轮数据格式

```json
{"text": "用户：你好\n助手：你好，有什么可以帮助你的吗？"}
```

### 多轮数据格式

```json
{"text": "<_user>你好<_bot>你好，有什么可以帮助你的吗？<_user>介绍一下你自己<_bot>我是TeleChat，一个大型语言模型。"}
```

### 数据处理命令

```bash
cd deepspeed-telechat/sft

# 处理数据
python process_data.py \
    --data_path data.json \
    --tokenizer_path ../../models/12B \
    --data_output_path datas/data_files \
    --max_seq_len 4096 \
    --num_samples 10000 \
    --num_workers 10 \
    --process_method multiple \
    --seed 42
```

---

## 服务部署

### API 服务

```bash
cd service

# 启动 API 服务
python telechat_service.py

# 指定模型和端口（需修改脚本）
# 编辑 telechat_service.py 修改以下变量：
# - MODEL_PATH: 模型路径
# - PORT: 服务端口（默认 8070）
```

### 测试 API

```bash
cd service

# 测试 JSON 接口
python test_json.py

# 测试流式接口
python test_stream.py

# 使用 curl 测试
curl -X POST "http://localhost:8070/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "你好", "max_length": 100}'
```

### Web 界面

```bash
cd service

# 启动 Web 界面
streamlit run web_demo.py

# 指定端口
streamlit run web_demo.py --server.port 8501

# 指定地址和端口
streamlit run web_demo.py --server.address 0.0.0.0 --server.port 8501
```

### Docker 部署

```bash
# 构建镜像（如果提供了 Dockerfile）
docker build -t telechat:latest .

# 运行容器
docker run -d \
    --name telechat \
    --gpus all \
    -p 8070:8070 \
    -p 8501:8501 \
    -v $(pwd)/models:/app/models \
    telechat:latest

# 查看日志
docker logs -f telechat

# 停止容器
docker stop telechat
```

---

## 常见问题

### GPU 内存不足

```bash
# 使用量化模型
python deploy.py --model-path models/7B-int4

# 减小 batch size
# 在训练脚本中设置：--per_device_train_batch_size 1

# 开启梯度检查点
# 在训练脚本中添加：--gradient_checkpointing

# 使用 DeepSpeed ZeRO-3
# 在训练脚本中设置：--zero_stage 3 --offload
```

### 端口被占用

```bash
# 查看端口占用
lsof -i :8070
netstat -tulpn | grep 8070

# 杀死占用进程
kill -9 <PID>

# 或使用其他端口
python deploy.py --api-port 8080 --web-port 8502
```

### 模型加载失败

```bash
# 检查模型文件
ls -lh models/7B/

# 必需文件：
# - config.json
# - pytorch_model.bin (或 model.safetensors)
# - tokenizer.model
# - tokenizer_config.json

# 重新下载模型
rm -rf models/7B
git clone https://huggingface.co/Tele-AI/Telechat-7B models/7B
```

### 依赖包冲突

```bash
# 创建新的虚拟环境
python -m venv telechat_env
source telechat_env/bin/activate  # Linux/Mac
# 或
telechat_env\Scripts\activate  # Windows

# 重新安装依赖
pip install -r requirements.txt
```

---

## 性能优化

### 推理加速

```python
# 使用 Flash Attention
model = AutoModelForCausalLM.from_pretrained(
    PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    use_flash_attention_2=True  # 需要安装 flash-attn
)

# 使用 BetterTransformer
# pip install optimum
from optimum.bettertransformer import BetterTransformer
model = BetterTransformer.transform(model)
```

### 训练加速

```bash
# 使用混合精度训练
--precision bf16  # 或 fp16

# 使用梯度累积
--gradient_accumulation_steps 4

# 使用 FlashAttention-2
# 在脚本中添加环境变量：export FLASH_ATTENTION=1
```

---

## 环境变量参考

```bash
# GPU 设置
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 模型路径
export MODEL_PATH=models/12B

# 服务端口
export API_PORT=8070
export WEB_PORT=8501

# DeepSpeed 设置
export MASTER_PORT=29500

# 日志级别
export TRANSFORMERS_VERBOSITY=info
```

---

## 更多资源

- **GitHub 仓库**: https://github.com/Tele-AI/Telechat
- **模型下载**: https://huggingface.co/Tele-AI
- **文档**: 查看 `docs/` 目录
- **示例代码**: 查看 `examples/` 目录
- **技术报告**: [TeleChat Technical Report](https://arxiv.org/abs/2401.03804)

---

## 许可证

使用 TeleChat 模型需遵循《TeleChat模型社区许可协议》。商业用途需发送邮件至 tele_ai@chinatelecom.cn 申请授权。
