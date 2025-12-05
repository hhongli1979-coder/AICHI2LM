#!/bin/bash
# TeleChat 生产环境部署脚本
# Production Deployment Script for TeleChat

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "未找到命令: $1"
        return 1
    fi
    return 0
}

# 检查 Docker 和 Docker Compose
check_docker() {
    print_info "检查 Docker 和 Docker Compose..."
    
    if ! check_command docker; then
        print_error "Docker 未安装，请先安装 Docker"
        print_info "安装指南: https://docs.docker.com/engine/install/"
        exit 1
    fi
    
    if ! check_command docker-compose && ! docker compose version &> /dev/null; then
        print_error "Docker Compose 未安装，请先安装 Docker Compose"
        print_info "安装指南: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    print_success "Docker 和 Docker Compose 检查通过"
}

# 检查 NVIDIA Docker 支持
check_nvidia_docker() {
    print_info "检查 GPU 支持..."
    
    if ! check_command nvidia-smi; then
        print_warning "nvidia-smi 未找到，GPU 可能不可用"
        return 1
    fi
    
    # 检查 Docker 是否支持 GPU（更高效的方法）
    if docker info 2>&1 | grep -q "Runtimes.*nvidia"; then
        print_success "GPU 支持检查通过"
        return 0
    else
        print_warning "NVIDIA Container Toolkit 未正确配置"
        print_info "安装指南: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        return 1
    fi
}

# 检查模型文件
check_models() {
    print_info "检查模型文件..."
    
    if [ ! -d "models" ]; then
        print_error "models 目录不存在"
        print_info "请创建 models 目录并下载模型文件"
        exit 1
    fi
    
    # 检查是否有模型文件
    if [ -z "$(ls -A models/)" ]; then
        print_warning "models 目录为空"
        print_info "请下载模型文件到 models 目录"
        read -p "是否继续？(y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "模型文件检查通过"
    fi
}

# 检查环境变量文件
check_env_file() {
    print_info "检查环境变量配置..."
    
    if [ ! -f ".env.production" ]; then
        print_warning ".env.production 文件不存在，将创建默认配置"
        cat > .env.production << EOF
# 模型配置
MODEL_PATH=/app/models/7B
CUDA_VISIBLE_DEVICES=0

# 服务端口
API_PORT=8070
WEB_PORT=8501

# 日志级别
LOG_LEVEL=INFO

# Python 配置
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
EOF
        print_success "已创建默认 .env.production 文件"
    fi
    
    if [ ! -f ".env" ]; then
        print_info "复制 .env.production 到 .env"
        cp .env.production .env
    fi
    
    print_success "环境变量配置检查通过"
}

# 构建 Docker 镜像
build_images() {
    print_info "构建 Docker 镜像..."
    
    if docker-compose build; then
        print_success "Docker 镜像构建成功"
    else
        print_error "Docker 镜像构建失败"
        exit 1
    fi
}

# 启动服务
start_services() {
    print_info "启动服务..."
    
    if docker-compose up -d; then
        print_success "服务启动成功"
    else
        print_error "服务启动失败"
        exit 1
    fi
}

# 等待服务就绪
wait_for_services() {
    print_info "等待服务就绪..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost/health > /dev/null 2>&1; then
            print_success "服务已就绪"
            return 0
        fi
        
        print_info "等待服务启动... ($attempt/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done
    
    print_error "服务启动超时"
    print_info "查看日志: docker-compose logs"
    return 1
}

# 显示服务状态
show_status() {
    print_info "服务状态:"
    docker-compose ps
    
    echo ""
    print_info "访问地址:"
    echo "  - API 文档: http://localhost/api/docs"
    echo "  - Web 界面: http://localhost/"
    echo "  - 健康检查: http://localhost/health"
    
    echo ""
    print_info "查看日志:"
    echo "  - TeleChat 日志: docker-compose logs -f telechat"
    echo "  - Nginx 日志: docker-compose logs -f nginx"
    echo "  - 所有日志: docker-compose logs -f"
    
    echo ""
    print_info "停止服务:"
    echo "  - docker-compose down"
}

# 主函数
main() {
    echo "============================================================"
    echo "🚀 TeleChat 生产环境部署"
    echo "============================================================"
    echo ""
    
    # 检查依赖
    check_docker
    check_nvidia_docker || print_warning "继续部署，但 GPU 可能不可用"
    check_models
    check_env_file
    
    echo ""
    
    # 确认部署
    print_warning "即将开始生产部署"
    read -p "确认继续？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "部署已取消"
        exit 0
    fi
    
    echo ""
    
    # 构建和启动
    build_images
    echo ""
    start_services
    echo ""
    wait_for_services
    
    echo ""
    echo "============================================================"
    print_success "✨ 部署成功！"
    echo "============================================================"
    echo ""
    show_status
}

# 运行主函数
main
