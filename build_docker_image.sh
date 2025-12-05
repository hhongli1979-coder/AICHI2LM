#!/bin/bash
# TeleChat Docker 镜像打包脚本
# 用于构建并保存 Docker 镜像为 tar 文件，方便分发和离线安装

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
IMAGE_NAME="telechat"
IMAGE_TAG="latest"
IMAGE_FULL="${IMAGE_NAME}:${IMAGE_TAG}"
OUTPUT_FILE="telechat-docker-image.tar"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}TeleChat Docker 镜像打包工具${NC}"
echo -e "${GREEN}========================================${NC}\n"

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker 未安装${NC}"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查 Dockerfile 是否存在
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}错误: 找不到 Dockerfile${NC}"
    echo "请确保在项目根目录执行此脚本"
    exit 1
fi

echo -e "${YELLOW}步骤 1/3: 构建 Docker 镜像...${NC}"
echo "镜像名称: ${IMAGE_FULL}"
echo ""

# 构建镜像
if docker build -t ${IMAGE_FULL} .; then
    echo -e "${GREEN}✓ 镜像构建成功${NC}\n"
else
    echo -e "${RED}✗ 镜像构建失败${NC}"
    exit 1
fi

echo -e "${YELLOW}步骤 2/3: 保存镜像到文件...${NC}"
echo "输出文件: ${OUTPUT_FILE}"
echo ""

# 保存镜像
if docker save -o ${OUTPUT_FILE} ${IMAGE_FULL}; then
    echo -e "${GREEN}✓ 镜像保存成功${NC}\n"
else
    echo -e "${RED}✗ 镜像保存失败${NC}"
    exit 1
fi

echo -e "${YELLOW}步骤 3/3: 压缩镜像文件...${NC}"
echo ""

# 压缩镜像文件
if command -v gzip &> /dev/null; then
    gzip -f ${OUTPUT_FILE}
    OUTPUT_FILE="${OUTPUT_FILE}.gz"
    echo -e "${GREEN}✓ 镜像压缩成功${NC}\n"
else
    echo -e "${YELLOW}! 跳过压缩（未安装 gzip）${NC}\n"
fi

# 获取文件大小
FILE_SIZE=$(du -h ${OUTPUT_FILE} | cut -f1)

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}打包完成！${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo "镜像信息:"
echo "  - 名称: ${IMAGE_FULL}"
echo "  - 文件: ${OUTPUT_FILE}"
echo "  - 大小: ${FILE_SIZE}"
echo ""

echo "使用说明:"
echo "  1. 将 ${OUTPUT_FILE} 传输到目标服务器"
if [[ ${OUTPUT_FILE} == *.gz ]]; then
    echo "  2. 解压: gunzip ${OUTPUT_FILE}"
    echo "  3. 加载: docker load -i ${OUTPUT_FILE%.gz}"
else
    echo "  2. 加载: docker load -i ${OUTPUT_FILE}"
fi
echo "  3. 运行: docker-compose up -d"
echo ""

echo -e "${YELLOW}注意事项:${NC}"
echo "  - 确保目标服务器已安装 Docker"
echo "  - GPU 支持需要安装 nvidia-container-toolkit"
echo "  - 需要准备模型文件到 models/ 目录"
echo ""
