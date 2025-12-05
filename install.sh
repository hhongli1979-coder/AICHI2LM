#!/bin/bash
# TeleChat Docker 一键安装脚本
# 自动完成所有安装步骤

set -e

echo "=========================================="
echo "TeleChat Docker 一键安装"
echo "=========================================="
echo ""

# 检测是否为 root 用户
if [ "$EUID" -eq 0 ]; then 
    SUDO=""
else
    SUDO="sudo"
fi

echo "步骤 1/5: 安装 Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    $SUDO sh /tmp/get-docker.sh
    $SUDO usermod -aG docker $USER
    echo "Docker 安装完成"
else
    echo "Docker 已安装，跳过"
fi
echo ""

echo "步骤 2/5: 安装 NVIDIA Container Toolkit（GPU 支持）..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    $SUDO apt-get update
    $SUDO apt-get install -y nvidia-container-toolkit
    $SUDO systemctl restart docker
    echo "NVIDIA Container Toolkit 安装完成"
else
    echo "NVIDIA Container Toolkit 已安装，跳过"
fi
echo ""

echo "步骤 3/5: 检查项目目录..."
if [ ! -f "docker-compose.yml" ]; then
    echo "错误：找不到 docker-compose.yml"
    echo "请确保在项目根目录（AICHI2LM）运行此脚本"
    exit 1
fi
echo "当前目录：$(pwd)"
echo ""

echo "步骤 4/5: 检查模型目录..."
if [ ! -d "models/7B" ]; then
    echo "创建模型目录..."
    mkdir -p models/7B
    echo "警告：请将模型文件复制到 models/7B 目录"
else
    echo "模型目录已存在"
fi
echo ""

echo "步骤 5/5: 启动服务..."
docker compose up -d
echo ""

echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "访问服务："
echo "  - API 文档: http://localhost:8070/docs"
echo "  - Web 界面: http://localhost:8501"
echo ""
echo "常用命令："
echo "  - 查看日志: docker compose logs -f"
echo "  - 停止服务: docker compose down"
echo "  - 重启服务: docker compose restart"
echo ""
