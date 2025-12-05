#!/bin/bash
# TeleChat 一键本地部署脚本 (One-Click Local Deployment Script)

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
MODEL_PATH="${MODEL_PATH:-../models/7B}"
API_PORT="${API_PORT:-8070}"
WEB_PORT="${WEB_PORT:-8501}"
GPU_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVICE_DIR="${SCRIPT_DIR}/service"

# PID文件
API_PID_FILE="/tmp/telechat_api.pid"
WEB_PID_FILE="/tmp/telechat_web.pid"

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ ${NC}$1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# 清理函数
cleanup() {
    print_info "正在停止服务..."
    
    # 停止Web服务
    if [ -f "$WEB_PID_FILE" ]; then
        WEB_PID=$(cat "$WEB_PID_FILE")
        if ps -p $WEB_PID > /dev/null 2>&1; then
            kill $WEB_PID 2>/dev/null
            print_success "Web服务已停止"
        fi
        rm -f "$WEB_PID_FILE"
    fi
    
    # 停止API服务
    if [ -f "$API_PID_FILE" ]; then
        API_PID=$(cat "$API_PID_FILE")
        if ps -p $API_PID > /dev/null 2>&1; then
            kill $API_PID 2>/dev/null
            print_success "API服务已停止"
        fi
        rm -f "$API_PID_FILE"
    fi
    
    exit 0
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "未找到命令: $1"
        return 1
    fi
    return 0
}

# 检查端口是否被占用
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        print_error "端口 $1 已被占用"
        return 1
    fi
    return 0
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖项..."
    
    # 检查Python
    if ! check_command python3; then
        return 1
    fi
    
    # 检查Python包
    python3 -c "import torch, transformers, fastapi, uvicorn, streamlit" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "缺少必要的Python包"
        print_info "请运行: pip install -r requirements.txt"
        return 1
    fi
    
    print_success "依赖项检查通过"
    return 0
}

# 检查模型路径
check_model() {
    print_info "检查模型路径: $MODEL_PATH"
    
    if [ ! -d "$MODEL_PATH" ]; then
        print_error "模型路径不存在: $MODEL_PATH"
        print_info "请设置正确的MODEL_PATH环境变量或修改脚本中的默认值"
        return 1
    fi
    
    # 检查config.json
    if [ ! -f "$MODEL_PATH/config.json" ]; then
        print_warning "未找到config.json，模型可能不完整"
    fi
    
    print_success "模型路径检查通过"
    return 0
}

# 等待API服务启动
wait_for_api() {
    print_info "等待API服务启动..."
    
    for i in {1..30}; do
        if curl -s http://127.0.0.1:${API_PORT}/docs > /dev/null 2>&1; then
            print_success "API服务已就绪"
            return 0
        fi
        sleep 2
    done
    
    print_error "API服务启动超时"
    return 1
}

# 启动API服务
start_api() {
    print_info "启动API服务 (端口: ${API_PORT})..."
    
    # 检查端口
    if ! check_port $API_PORT; then
        return 1
    fi
    
    # 启动服务
    cd "$SERVICE_DIR"
    export CUDA_VISIBLE_DEVICES=$GPU_DEVICES
    nohup python3 telechat_service.py > /tmp/telechat_api.log 2>&1 &
    API_PID=$!
    echo $API_PID > "$API_PID_FILE"
    
    # 等待服务启动
    if ! wait_for_api; then
        kill $API_PID 2>/dev/null
        rm -f "$API_PID_FILE"
        return 1
    fi
    
    print_success "API服务已启动 (PID: $API_PID)"
    print_info "API文档: http://0.0.0.0:${API_PORT}/docs"
    return 0
}

# 启动Web服务
start_web() {
    print_info "启动Web服务 (端口: ${WEB_PORT})..."
    
    # 检查端口
    if ! check_port $WEB_PORT; then
        return 1
    fi
    
    # 启动服务
    cd "$SERVICE_DIR"
    nohup streamlit run web_demo.py --server.port $WEB_PORT --server.address 0.0.0.0 > /tmp/telechat_web.log 2>&1 &
    WEB_PID=$!
    echo $WEB_PID > "$WEB_PID_FILE"
    
    # 等待服务启动
    sleep 5
    
    if ! ps -p $WEB_PID > /dev/null 2>&1; then
        print_error "Web服务启动失败"
        rm -f "$WEB_PID_FILE"
        return 1
    fi
    
    print_success "Web服务已启动 (PID: $WEB_PID)"
    print_info "Web界面: http://0.0.0.0:${WEB_PORT}"
    return 0
}

# 显示使用说明
show_usage() {
    cat << EOF
TeleChat 一键本地部署脚本

使用方法:
  $0 [选项]

选项:
  -h, --help              显示帮助信息
  -m, --model PATH        指定模型路径 (默认: ../models/7B)
  -g, --gpu DEVICES       指定GPU设备 (默认: 0)
  -a, --api-port PORT     指定API端口 (默认: 8070)
  -w, --web-port PORT     指定Web端口 (默认: 8501)

环境变量:
  MODEL_PATH              模型路径
  CUDA_VISIBLE_DEVICES    GPU设备
  API_PORT                API服务端口
  WEB_PORT                Web服务端口

示例:
  # 使用默认配置
  $0

  # 指定模型路径
  $0 --model /path/to/model

  # 指定GPU设备
  $0 --gpu 0,1

  # 指定端口
  $0 --api-port 8080 --web-port 8502
EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_DEVICES="$2"
            shift 2
            ;;
        -a|--api-port)
            API_PORT="$2"
            shift 2
            ;;
        -w|--web-port)
            WEB_PORT="$2"
            shift 2
            ;;
        *)
            print_error "未知选项: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 主函数
main() {
    echo "============================================================"
    echo "🎯 TeleChat 一键本地部署"
    echo "============================================================"
    echo ""
    
    # 检查依赖
    if ! check_dependencies; then
        exit 1
    fi
    
    # 检查模型
    if ! check_model; then
        exit 1
    fi
    
    echo ""
    
    # 启动API服务
    if ! start_api; then
        cleanup
        exit 1
    fi
    
    echo ""
    
    # 启动Web服务
    if ! start_web; then
        cleanup
        exit 1
    fi
    
    echo ""
    echo "============================================================"
    echo "✨ 部署成功！"
    echo "============================================================"
    echo "📍 API服务: http://0.0.0.0:${API_PORT}/docs"
    echo "📍 Web界面: http://0.0.0.0:${WEB_PORT}"
    echo ""
    echo "日志文件:"
    echo "  API: /tmp/telechat_api.log"
    echo "  Web: /tmp/telechat_web.log"
    echo ""
    echo "按 Ctrl+C 停止服务"
    echo "============================================================"
    
    # 保持运行
    while true; do
        sleep 1
        
        # 检查进程是否仍在运行
        if [ -f "$API_PID_FILE" ]; then
            API_PID=$(cat "$API_PID_FILE")
            if ! ps -p $API_PID > /dev/null 2>&1; then
                print_error "API服务意外停止"
                cleanup
                exit 1
            fi
        fi
        
        if [ -f "$WEB_PID_FILE" ]; then
            WEB_PID=$(cat "$WEB_PID_FILE")
            if ! ps -p $WEB_PID > /dev/null 2>&1; then
                print_error "Web服务意外停止"
                cleanup
                exit 1
            fi
        fi
    done
}

# 运行主函数
main
