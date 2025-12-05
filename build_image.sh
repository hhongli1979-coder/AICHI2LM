#!/bin/bash
# TeleChat Docker Image Build Script
# 构建 TeleChat Docker 镜像的辅助脚本

set -e

REGISTRY_URL="${REGISTRY_URL:-localhost:5000}"
IMAGE_NAME="${IMAGE_NAME:-telechat}"

echo "=== TeleChat Docker 镜像构建脚本 ==="
echo ""

# 检查参数
if [ "$#" -lt 1 ]; then
    echo "用法: $0 <模型版本> [其他选项]"
    echo ""
    echo "模型版本:"
    echo "  7b-fp16    - TeleChat 7B FP16 精度"
    echo "  7b-int8    - TeleChat 7B INT8 量化"
    echo "  7b-int4    - TeleChat 7B INT4 量化"
    echo "  12b-fp16   - TeleChat 12B FP16 精度"
    echo "  12b-int8   - TeleChat 12B INT8 量化"
    echo "  12b-int4   - TeleChat 12B INT4 量化"
    echo ""
    echo "示例:"
    echo "  $0 7b-fp16              # 构建 7B FP16 镜像"
    echo "  $0 12b-fp16 --no-cache  # 构建 12B FP16 镜像 (不使用缓存)"
    exit 1
fi

MODEL_VERSION="$1"
shift

# 构建标签
TAG="${REGISTRY_URL}/${IMAGE_NAME}:${MODEL_VERSION}"
LATEST_TAG="${REGISTRY_URL}/${IMAGE_NAME}:latest"

echo "构建信息:"
echo "  - 仓库地址: ${REGISTRY_URL}"
echo "  - 镜像名称: ${IMAGE_NAME}"
echo "  - 模型版本: ${MODEL_VERSION}"
echo "  - 完整标签: ${TAG}"
echo ""

# 检查 Dockerfile 是否存在
if [ ! -f "Dockerfile" ]; then
    echo "错误: 未找到 Dockerfile"
    exit 1
fi

# 构建镜像
echo "开始构建 Docker 镜像..."
docker build -t "${TAG}" "$@" .

# 添加 latest 标签 (如果构建的是 12b-fp16)
if [ "${MODEL_VERSION}" = "12b-fp16" ]; then
    echo ""
    echo "为 12B FP16 版本添加 latest 标签..."
    docker tag "${TAG}" "${LATEST_TAG}"
fi

echo ""
echo "=== 构建完成! ==="
echo ""
echo "镜像标签:"
echo "  ${TAG}"
if [ "${MODEL_VERSION}" = "12b-fp16" ]; then
    echo "  ${LATEST_TAG}"
fi
echo ""

# 询问是否推送到仓库
read -p "是否推送镜像到仓库? (y/N): " push_confirm
if [[ "$push_confirm" =~ ^[Yy]$ ]]; then
    echo ""
    echo "推送镜像到仓库..."
    docker push "${TAG}"
    
    if [ "${MODEL_VERSION}" = "12b-fp16" ]; then
        docker push "${LATEST_TAG}"
    fi
    
    echo ""
    echo "=== 推送完成! ==="
    echo ""
    echo "注册镜像到访问控制系统..."
    
    # 获取镜像大小和创建时间
    IMAGE_SIZE=$(docker image inspect "${TAG}" --format='{{.Size}}' 2>/dev/null || echo "0")
    IMAGE_CREATED=$(docker image inspect "${TAG}" --format='{{.Created}}' 2>/dev/null || echo "")
    
    if [ -n "$IMAGE_SIZE" ] && [ "$IMAGE_SIZE" != "0" ]; then
        python3 registry_cli.py image register "${IMAGE_NAME}" "${MODEL_VERSION}" \
            --size "$IMAGE_SIZE" \
            --created "$IMAGE_CREATED"
    else
        echo "警告: 无法获取镜像元数据，跳过自动注册"
        echo "请手动注册镜像: python3 registry_cli.py image register ${IMAGE_NAME} ${MODEL_VERSION}"
    fi
    
    echo ""
    echo "镜像已推送并注册。"
    echo ""
    echo "后续操作:"
    echo "  # 设置镜像访问权限"
    echo "  python3 registry_cli.py access grant ${IMAGE_NAME} ${MODEL_VERSION} <username>"
    echo ""
    echo "  # 其他用户拉取镜像"
    echo "  docker login ${REGISTRY_URL}"
    echo "  docker pull ${TAG}"
else
    echo ""
    echo "镜像已构建但未推送。"
    echo ""
    echo "手动推送:"
    echo "  docker push ${TAG}"
fi

echo ""
