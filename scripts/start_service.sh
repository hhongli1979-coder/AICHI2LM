#!/bin/bash
# TeleChat API 服务启动脚本

set -e

# 默认配置
HOST=${TELECHAT_HOST:-"0.0.0.0"}
PORT=${TELECHAT_PORT:-8080}
WORKERS=${TELECHAT_WORKERS:-1}
CONFIG_PATH=${TELECHAT_CONFIG:-"configs/config.yaml"}

# 显示配置
echo "=========================================="
echo "TeleChat API Service"
echo "=========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "Config: $CONFIG_PATH"
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found"
    exit 1
fi

# 创建日志目录
mkdir -p logs

# 启动服务
echo "Starting TeleChat API service..."

if [ "$1" == "--dev" ]; then
    # 开发模式（带热重载）
    python3 -m uvicorn telechat.api.main:app \
        --host $HOST \
        --port $PORT \
        --reload \
        --reload-dir src
else
    # 生产模式
    python3 -m uvicorn telechat.api.main:app \
        --host $HOST \
        --port $PORT \
        --workers $WORKERS
fi
