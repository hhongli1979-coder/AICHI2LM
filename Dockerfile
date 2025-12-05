# TeleChat Production Dockerfile
# 星辰语义大模型生产环境部署镜像

# 使用 NVIDIA CUDA 基础镜像
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES=0

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 创建软链接
RUN ln -s /usr/bin/python3 /usr/bin/python

# 升级 pip
RUN pip3 install --upgrade pip setuptools wheel

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（分步安装以优化缓存）
RUN pip3 install torch==1.13.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install transformers==4.30.0 accelerate>=0.24.1
RUN pip3 install fastapi>=0.109.1 uvicorn==0.17.6 streamlit>=1.30.0
RUN pip3 install pyyaml>=6.0 psutil>=5.9.0 requests>=2.28.0
RUN pip3 install deepspeed==0.8.3 datasets>=2.10.1 peft>=0.5.0
RUN pip3 install jsonlines safetensors>=0.3.1

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p /app/logs /app/models

# 暴露端口
EXPOSE 8070 8501

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8070/health || exit 1

# 设置启动命令
CMD ["python", "deploy.py"]
