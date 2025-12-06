# TeleChat 安装指南

本文档提供TeleChat大模型的详细安装说明，包括各种环境和场景下的安装方法。

## 目录

- [快速安装](#快速安装)
- [详细安装步骤](#详细安装步骤)
- [Docker安装](#docker安装)
- [从源码安装](#从源码安装)
- [国产化适配环境](#国产化适配环境)
- [常见问题](#常见问题)

---

## 快速安装

如果您熟悉Python环境配置，可以直接执行以下命令：

```bash
# 1. 创建虚拟环境
conda create -n telechat python=3.9 -y
conda activate telechat

# 2. 安装PyTorch (CUDA 11.7)
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 3. 克隆仓库并安装依赖
git clone https://github.com/Tele-AI/TeleChat.git
cd TeleChat
pip install -r requirements.txt

# 4. 可选: 安装FlashAttention2
pip install flash-attn --no-build-isolation

# 5. 验证安装
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 详细安装步骤

### 1. 系统要求

#### 硬件要求

| 用途 | GPU | 内存 | 存储 |
|-----|-----|-----|------|
| 推理(7B) | NVIDIA GPU 8GB+ | 16GB+ | 30GB+ |
| 推理(12B) | NVIDIA GPU 16GB+ | 32GB+ | 50GB+ |
| 训练(7B) | 8x A100 40GB | 256GB+ | 100GB+ |
| 训练(12B) | 8x A100 40GB | 512GB+ | 200GB+ |

#### 软件要求

| 软件 | 版本要求 | 说明 |
|-----|---------|------|
| 操作系统 | Linux/macOS/Windows | 推荐Ubuntu 18.04+ |
| Python | 3.8 / 3.9 / 3.10 | 推荐3.9 |
| CUDA | 11.6 / 11.7 / 11.8 | GPU版本必需 |
| cuDNN | 8.x | 与CUDA版本匹配 |
| Git | 2.x | 克隆代码需要 |
| Git LFS | latest | 下载模型需要 |

### 2. 准备Python环境

#### 使用Conda (推荐)

```bash
# 安装Miniconda (如果尚未安装)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建并激活环境
conda create -n telechat python=3.9 -y
conda activate telechat

# 验证Python版本
python --version  # 应显示 Python 3.9.x
```

#### 使用venv

```bash
# 创建虚拟环境
python3.9 -m venv telechat_env

# 激活环境
source telechat_env/bin/activate  # Linux/macOS
# telechat_env\Scripts\activate   # Windows

# 升级pip
pip install --upgrade pip
```

### 3. 安装PyTorch

PyTorch版本需要与CUDA版本匹配。首先检查您的CUDA版本：

```bash
# 检查CUDA版本
nvcc --version

# 或
nvidia-smi
```

然后根据CUDA版本安装对应的PyTorch：

#### CUDA 11.6

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

#### CUDA 11.7

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

#### CUDA 11.8

```bash
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 12.1

```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

#### CPU版本 (不推荐用于训练)

```bash
pip install torch==1.13.1 torchvision==0.14.1
```

#### 验证PyTorch安装

```bash
python << EOF
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
EOF
```

### 4. 克隆TeleChat仓库

```bash
# 克隆代码
git clone https://github.com/Tele-AI/TeleChat.git
cd TeleChat

# 查看目录结构
ls -la
```

### 5. 安装项目依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 如果安装过程中遇到问题，可以尝试逐个安装
pip install accelerate>=0.24.1
pip install auto-gptq==0.3.0
pip install deepspeed==0.8.3
pip install datasets>=2.10.1
pip install jsonlines
pip install peft>=0.5.0
pip install safetensors>=0.3.1
pip install transformers==4.30.0
pip install fastapi>=0.109.1
pip install uvicorn==0.17.6
pip install streamlit>=1.30.0
pip install pyyaml>=6.0
pip install psutil>=5.9.0
pip install requests>=2.28.0
```

### 6. 安装FlashAttention2 (可选)

FlashAttention2可以显著提升性能，但安装较为复杂：

```bash
# 方式1: 使用pip安装 (需要编译)
pip install flash-attn --no-build-isolation

# 方式2: 从源码安装
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install .

# 方式3: 使用预编译wheel (推荐)
# 访问 https://github.com/Dao-AILab/flash-attention/releases
# 下载对应的wheel文件后安装
pip install flash_attn-2.x.x+cuXXX-cpXX-cpXX-linux_x86_64.whl
```

**注意事项**:
- FlashAttention2需要CUDA 11.6+
- 需要安装CUDA开发工具包
- 编译可能需要10-30分钟
- 如果安装失败，可以跳过此步骤

### 7. 验证安装

创建测试脚本 `test_install.py`:

```python
#!/usr/bin/env python3
import sys

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name}: Not installed")
        return False

print("检查TeleChat依赖安装情况...\n")

# 核心依赖
print("核心依赖:")
check_package("torch")
check_package("transformers")
check_package("deepspeed")
check_package("accelerate")

# 训练相关
print("\n训练相关:")
check_package("peft")
check_package("datasets")

# 推理服务
print("\n推理服务:")
check_package("fastapi")
check_package("uvicorn")
check_package("streamlit")

# 量化
print("\n量化工具:")
check_package("auto-gptq", "auto_gptq")

# 可选依赖
print("\n可选依赖:")
try:
    import flash_attn
    print(f"✓ flash-attn: {flash_attn.__version__}")
except:
    print("✗ flash-attn: Not installed (可选)")

# 检查CUDA
print("\nCUDA支持:")
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
```

运行验证：

```bash
python test_install.py
```

---

## Docker安装

使用Docker可以避免环境配置问题，推荐用于快速体验。

### 1. 安装Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. 下载预构建镜像

```bash
# 从天翼云盘下载
# 下载地址: https://cloud.189.cn/web/share?code=vQFJRf7JBfmq (访问码: ona6)

# 导入镜像
sudo docker load -i telechat-public_1.2.tar

# 验证镜像
sudo docker images | grep telechat
```

### 3. 启动容器

```bash
# 启动容器 (使用所有GPU)
sudo docker run -itd \
  --name telechat \
  --runtime=nvidia \
  --gpus all \
  --shm-size=256g \
  -v /path/to/your/data:/data \
  -v /path/to/your/models:/models \
  -p 8070:8070 \
  -p 8501:8501 \
  telechat-public:1.2 \
  bash

# 进入容器
sudo docker exec -it telechat bash

# 在容器内验证环境
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 4. 使用Dockerfile构建

如果需要自定义镜像，可以使用以下Dockerfile:

```dockerfile
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装Python和基础工具
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git \
    git-lfs \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安装PyTorch
RUN pip3 install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 克隆TeleChat
WORKDIR /workspace
RUN git clone https://github.com/Tele-AI/TeleChat.git

# 安装依赖
WORKDIR /workspace/TeleChat
RUN pip3 install -r requirements.txt

# 设置工作目录
WORKDIR /workspace/TeleChat

CMD ["/bin/bash"]
```

构建并运行:

```bash
# 构建镜像
docker build -t telechat:custom .

# 运行容器
docker run -itd --name telechat --runtime=nvidia --gpus all telechat:custom
```

---

## 从源码安装

如果需要修改源码或使用最新开发版本：

```bash
# 1. 克隆仓库
git clone https://github.com/Tele-AI/TeleChat.git
cd TeleChat

# 2. 创建开发环境
conda create -n telechat-dev python=3.9
conda activate telechat-dev

# 3. 安装编辑模式
pip install -e .

# 4. 安装开发依赖
pip install pytest black flake8 mypy

# 5. 运行测试
pytest tests/
```

---

## 国产化适配环境

### 昇腾NPU环境

TeleChat支持华为昇腾NPU，详细说明：

#### 昇腾Atlas 300I Pro推理卡

```bash
# 1. 安装昇腾驱动和CANN
# 参考: https://www.hiascend.com/

# 2. 安装MindSpore或PyTorch NPU版本
pip install torch-npu

# 3. 验证NPU
python -c "import torch_npu; print('NPU available:', torch_npu.npu.is_available())"

# 4. 克隆TeleChat NPU适配代码
git clone https://gitee.com/ascend/ModelLink.git
cd ModelLink/mindie_ref/mindie_llm/atb_models/pytorch/examples/telechat
```

#### 昇腾Atlas 800T A2训练服务器

**使用MindSpore**:

```bash
# 1. 安装MindSpore
pip install mindspore==2.2.0

# 2. 克隆MindFormers
git clone https://gitee.com/mindspore/mindformers.git
cd mindformers/research/telechat

# 3. 按照README配置和训练
```

**使用PyTorch**:

```bash
# 1. 安装PyTorch NPU版本
pip install torch-npu

# 2. 克隆ModelZoo-PyTorch
git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
cd ModelZoo-PyTorch/PyTorch/contrib/nlp/Telechat

# 3. 按照README配置和训练
```

---

## 常见问题

### Q1: pip安装速度慢

**解决方案**: 使用国内镜像源

```bash
# 临时使用
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name

# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: CUDA版本不匹配

**症状**: 
```
RuntimeError: CUDA error: no kernel image is available for execution
```

**解决方案**:
```bash
# 检查CUDA版本
nvcc --version
nvidia-smi  # 查看Driver版本

# 重新安装匹配的PyTorch
pip uninstall torch torchvision
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Q3: 显存不足

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
1. 使用量化模型 (int8/int4)
2. 减小batch size
3. 使用gradient checkpointing:
   ```python
   model.gradient_checkpointing_enable()
   ```
4. 使用CPU offload:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_path, 
       device_map="auto",
       offload_folder="offload"
   )
   ```

### Q4: FlashAttention2安装失败

**症状**: 编译错误

**解决方案**:
```bash
# 1. 确保安装了CUDA开发工具
sudo apt-get install cuda-toolkit-11-7

# 2. 设置环境变量
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 3. 重试安装
pip install flash-attn --no-build-isolation

# 4. 如果还是失败，跳过FlashAttention2
# 模型会自动使用标准注意力机制
```

### Q5: transformers版本冲突

**症状**: 
```
ImportError: cannot import name 'XXX' from 'transformers'
```

**解决方案**:
```bash
# 严格安装指定版本
pip uninstall transformers -y
pip install transformers==4.30.0 --no-deps
pip install -r requirements.txt
```

### Q6: deepspeed安装失败

**症状**: 编译错误或找不到ninja

**解决方案**:
```bash
# 安装依赖
pip install ninja

# 如果在编译ops时失败，可以跳过预编译
DS_BUILD_OPS=0 pip install deepspeed==0.8.3
```

### Q7: Windows系统安装问题

**解决方案**:

方案1: 安装Microsoft C++ Build Tools
```
下载: https://visualstudio.microsoft.com/visual-cpp-build-tools/
安装: Desktop development with C++
```

方案2: 使用Anaconda
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

方案3: 使用WSL2
```bash
# 在PowerShell中以管理员身份运行
wsl --install
wsl --set-default-version 2

# 在WSL2 Ubuntu中按Linux方式安装
```

### Q8: Git LFS下载模型失败

**症状**: 
```
Error downloading object: ...
```

**解决方案**:
```bash
# 安装git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# 初始化
git lfs install

# 重新克隆
git lfs clone https://huggingface.co/Tele-AI/Telechat-7B

# 或使用huggingface-cli
pip install huggingface-hub
huggingface-cli download Tele-AI/Telechat-7B --local-dir ./models/7B
```

---

## 获取帮助

如果遇到本文档未涵盖的问题:

1. 查看项目GitHub Issues: https://github.com/Tele-AI/TeleChat/issues
2. 查阅相关文档:
   - [README.md](./README.md)
   - [tutorial.md](./docs/tutorial.md)
   - [DEPLOYMENT.md](./DEPLOYMENT.md)
3. 提交新的Issue并提供:
   - 操作系统和版本
   - Python版本
   - CUDA版本
   - 完整的错误信息
   - 已尝试的解决方案

---

## 更新日志

- 2024-12: 初始版本，包含基础安装说明和常见问题
