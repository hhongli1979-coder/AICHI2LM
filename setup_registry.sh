#!/bin/bash
# TeleChat Docker Registry Setup Script
# 初始化和启动 Docker 镜像仓库

set -e

echo "=== TeleChat Docker Registry 设置脚本 ==="
echo ""

# 检查 Docker 是否已安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker 未安装"
    echo "请访问 https://docs.docker.com/get-docker/ 安装 Docker"
    exit 1
fi

# 检查 Docker Compose 是否已安装
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "错误: Docker Compose 未安装"
    echo "请访问 https://docs.docker.com/compose/install/ 安装 Docker Compose"
    exit 1
fi

# 检查 htpasswd 是否已安装
if ! command -v htpasswd &> /dev/null; then
    echo "警告: htpasswd 未安装"
    echo "正在尝试安装..."
    
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y apache2-utils
    elif command -v yum &> /dev/null; then
        sudo yum install -y httpd-tools
    else
        echo "错误: 无法自动安装 htpasswd"
        echo "请手动安装 apache2-utils (Debian/Ubuntu) 或 httpd-tools (RHEL/CentOS)"
        exit 1
    fi
fi

# 创建必要的目录
echo "创建目录结构..."
mkdir -p registry/data
mkdir -p registry/auth
mkdir -p models

# 设置 Python CLI 工具为可执行
chmod +x registry_cli.py

echo ""
echo "=== 初始化管理员用户 ==="
read -p "请输入管理员用户名 (默认: admin): " admin_user
admin_user=${admin_user:-admin}

read -sp "请输入管理员密码 (留空将生成随机密码): " admin_pass
echo ""

# 如果未提供密码，生成一个强随机密码
if [ -z "$admin_pass" ]; then
    admin_pass=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-16)
    echo ""
    echo "⚠️  已生成随机密码，请妥善保存！" >&2
    echo "管理员密码: $admin_pass" >&2
    echo ""
fi

# 使用 Python CLI 工具创建管理员用户
echo "创建管理员用户..."
python3 registry_cli.py user add "$admin_user" "$admin_pass" --permissions admin --email "admin@example.com"

echo ""
echo "=== 启动 Docker Registry ==="
echo "正在启动服务..."

# 使用 docker-compose 启动服务
if command -v docker-compose &> /dev/null; then
    docker-compose up -d registry registry-ui
else
    docker compose up -d registry registry-ui
fi

echo ""
echo "=== 设置完成! ==="
echo ""
echo "Docker Registry 已启动，访问地址:"
echo "  - Registry API: http://localhost:5000"
echo "  - Registry UI:  http://localhost:8080"
echo ""
echo "管理员账户:"
echo "  - 用户名: $admin_user"
echo "  - 密码: $admin_pass"
echo ""
echo "使用示例:"
echo "  # 登录到仓库"
echo "  docker login localhost:5000 -u $admin_user -p $admin_pass"
echo ""
echo "  # 构建镜像"
echo "  docker build -t localhost:5000/telechat:7b-fp16 ."
echo ""
echo "  # 推送镜像"
echo "  docker push localhost:5000/telechat:7b-fp16"
echo ""
echo "  # 拉取镜像"
echo "  docker pull localhost:5000/telechat:7b-fp16"
echo ""
echo "管理命令:"
echo "  # 添加用户"
echo "  python3 registry_cli.py user add <username> <password> --permissions view,pull"
echo ""
echo "  # 列出所有用户"
echo "  python3 registry_cli.py user list"
echo ""
echo "  # 注册镜像并设置访问权限"
echo "  python3 registry_cli.py image register telechat 7b-fp16 --users user1,user2"
echo ""
echo "  # 检查访问权限"
echo "  python3 registry_cli.py access check user1 telechat 7b-fp16"
echo ""
