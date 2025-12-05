# TeleChat Docker 镜像仓库管理系统

## 概述

TeleChat Docker 镜像仓库管理系统是一个用于管理 Docker 镜像并控制访问权限的完整解决方案。该系统允许管理员：

- 🔐 创建和管理用户账户
- 📦 注册和管理 Docker 镜像
- 🔑 控制用户对特定镜像的访问权限
- 🖥️ 通过 Web UI 可视化管理镜像
- 📊 跟踪镜像元数据和访问记录

## 系统架构

系统由以下组件构成：

1. **Docker Registry** - 标准的 Docker 镜像仓库，支持基于 htpasswd 的认证
2. **Registry UI** - 可视化的 Web 界面，用于浏览和管理镜像
3. **访问控制管理器** - Python 实现的细粒度权限控制系统
4. **命令行工具** - 用于用户和权限管理的 CLI 工具

## 快速开始

### 前置要求

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.8+
- apache2-utils (提供 htpasswd 命令)

### 安装步骤

1. **克隆仓库**
```bash
git clone <your-repository-url>
cd AICHI2LM
```

2. **运行设置脚本**
```bash
chmod +x setup_registry.sh
./setup_registry.sh
```

该脚本会：
- 检查必要的依赖
- 创建目录结构
- 初始化管理员账户
- 启动 Docker Registry 和 UI

3. **访问系统**

启动后，您可以访问：
- **Registry API**: http://localhost:5000
- **Registry UI**: http://localhost:8080

## 用户管理

### 添加用户

```bash
python3 registry_cli.py user add <username> <password> --permissions <权限列表> --email <邮箱>
```

**权限类型:**
- `view` - 查看镜像列表
- `pull` - 拉取镜像
- `push` - 推送镜像
- `delete` - 删除镜像
- `admin` - 管理员权限 (包含所有权限)

**示例:**
```bash
# 添加管理员 (请使用强密码)
python3 registry_cli.py user add admin <STRONG_PASSWORD> --permissions admin --email admin@example.com

# 添加只读用户 (请使用强密码)
python3 registry_cli.py user add user1 <STRONG_PASSWORD> --permissions view,pull --email user1@example.com

# 添加可以推送的用户 (请使用强密码)
python3 registry_cli.py user add developer <STRONG_PASSWORD> --permissions view,pull,push --email dev@example.com
```

**安全提示:**
- 始终使用强密码（至少 12 个字符，包含大小写字母、数字和特殊字符）
- 不要在命令历史中保留密码，考虑使用密码管理器
- 定期更新密码

### 列出所有用户

```bash
python3 registry_cli.py user list
```

### 更新用户权限

```bash
python3 registry_cli.py user update <username> --permissions <新权限列表>
```

**示例:**
```bash
python3 registry_cli.py user update user1 --permissions view,pull,push
```

### 删除用户

```bash
python3 registry_cli.py user remove <username>
```

## 镜像管理

### 构建镜像

使用提供的脚本构建 TeleChat 镜像：

```bash
chmod +x build_image.sh
./build_image.sh <模型版本>
```

**支持的模型版本:**
- `7b-fp16` - TeleChat 7B FP16 精度
- `7b-int8` - TeleChat 7B INT8 量化
- `7b-int4` - TeleChat 7B INT4 量化
- `12b-fp16` - TeleChat 12B FP16 精度
- `12b-int8` - TeleChat 12B INT8 量化
- `12b-int4` - TeleChat 12B INT4 量化

**示例:**
```bash
# 构建 7B FP16 版本
./build_image.sh 7b-fp16

# 构建 12B FP16 版本 (不使用缓存)
./build_image.sh 12b-fp16 --no-cache
```

### 手动构建和推送

```bash
# 登录到仓库
docker login localhost:5000

# 构建镜像
docker build -t localhost:5000/telechat:7b-fp16 .

# 推送镜像
docker push localhost:5000/telechat:7b-fp16
```

### 注册镜像元数据

推送镜像后，需要在访问控制系统中注册：

```bash
python3 registry_cli.py image register <镜像名> <标签> --users <用户列表>
```

**参数说明:**
- `--users` - 允许访问的用户列表 (逗号分隔)，留空表示所有用户可访问
- `--digest` - 镜像摘要
- `--created` - 创建时间
- `--size` - 镜像大小 (字节)

**示例:**
```bash
# 只允许特定用户访问
python3 registry_cli.py image register telechat 7b-fp16 --users user1,user2

# 允许所有用户访问
python3 registry_cli.py image register telechat 7b-int4

# 包含完整元数据
python3 registry_cli.py image register telechat 12b-fp16 \
  --users admin,developer \
  --size 24000000000 \
  --created "2024-12-05T12:00:00Z"
```

### 列出用户可访问的镜像

```bash
python3 registry_cli.py image list-accessible <username>
```

**示例:**
```bash
python3 registry_cli.py image list-accessible user1
```

## 访问控制

### 授予访问权限

```bash
python3 registry_cli.py access grant <镜像名> <标签> <username>
```

**示例:**
```bash
# 授予 user3 访问 telechat:7b-fp16 的权限
python3 registry_cli.py access grant telechat 7b-fp16 user3
```

### 撤销访问权限

```bash
python3 registry_cli.py access revoke <镜像名> <标签> <username>
```

**示例:**
```bash
# 撤销 user2 访问 telechat:12b-fp16 的权限
python3 registry_cli.py access revoke telechat 12b-fp16 user2
```

### 检查访问权限

```bash
python3 registry_cli.py access check <username> <镜像名> <标签>
```

**示例:**
```bash
# 检查 user1 是否可以访问 telechat:7b-fp16
python3 registry_cli.py access check user1 telechat 7b-fp16
```

## 使用 Docker 镜像

### 普通用户拉取镜像

```bash
# 登录到仓库
docker login localhost:5000
# 输入用户名和密码

# 拉取镜像
docker pull localhost:5000/telechat:7b-fp16

# 运行容器
docker run --gpus all -it localhost:5000/telechat:7b-fp16 bash
```

### 运行 TeleChat 服务

使用 docker-compose 启动完整的 TeleChat API 服务：

```bash
docker-compose up -d telechat-api
```

服务将在 `http://localhost:8000` 上可用。

## Python API

您也可以在 Python 代码中直接使用访问控制管理器：

```python
from docker_registry_manager import DockerRegistryManager, Permission

# 创建管理器实例
manager = DockerRegistryManager(
    registry_url="localhost:5000",
    config_dir="./registry"
)

# 添加用户
manager.add_user("user1", "pass123", [Permission.VIEW.value, Permission.PULL.value])

# 注册镜像
manager.register_image("telechat", "7b-fp16", allowed_users=["user1"])

# 检查访问权限
has_access = manager.check_image_access("user1", "telechat", "7b-fp16")
print(f"用户有访问权限: {has_access}")

# 列出用户可访问的镜像
images = manager.list_accessible_images("user1")
for img in images:
    print(f"{img.name}:{img.tag}")

# 授予访问权限
manager.grant_image_access("telechat", "7b-fp16", "user2")

# 撤销访问权限
manager.revoke_image_access("telechat", "7b-fp16", "user2")
```

## 高级配置

### 自定义仓库地址

```bash
export REGISTRY_URL="myregistry.example.com:5000"
python3 registry_cli.py --registry-url $REGISTRY_URL user list
```

### 自定义配置目录

```bash
python3 registry_cli.py --config-dir /path/to/config user list
```

### 使用外部 Registry

如果您已经有一个运行中的 Docker Registry，可以只使用访问控制管理器：

```python
manager = DockerRegistryManager(
    registry_url="your-registry.example.com:5000",
    config_dir="/path/to/config"
)
```

## 配置文件

系统使用以下配置文件：

- `registry/users.json` - 用户信息和权限
- `registry/images.json` - 镜像元数据和访问控制
- `registry/auth/htpasswd` - Docker Registry 认证文件

**用户配置示例 (registry/users.json):**
```json
{
  "admin": {
    "username": "admin",
    "password_hash": "htpasswd",
    "permissions": ["admin"],
    "email": "admin@example.com"
  },
  "user1": {
    "username": "user1",
    "password_hash": "htpasswd",
    "permissions": ["view", "pull"],
    "email": "user1@example.com"
  }
}
```

**镜像配置示例 (registry/images.json):**
```json
{
  "telechat:7b-fp16": {
    "name": "telechat",
    "tag": "7b-fp16",
    "digest": "sha256:abc123...",
    "created": "2024-12-05T12:00:00Z",
    "size": 14000000000,
    "allowed_users": ["user1", "user2"]
  },
  "telechat:7b-int4": {
    "name": "telechat",
    "tag": "7b-int4",
    "digest": "sha256:def456...",
    "created": "2024-12-05T13:00:00Z",
    "size": 3500000000,
    "allowed_users": []
  }
}
```

## 故障排除

### htpasswd 命令未找到

**问题:** 运行脚本时提示 "htpasswd: command not found"

**解决方案:**
```bash
# Debian/Ubuntu
sudo apt-get install apache2-utils

# RHEL/CentOS
sudo yum install httpd-tools
```

### Docker Registry 无法启动

**问题:** Registry 容器启动失败

**解决方案:**
1. 检查端口 5000 是否被占用：
   ```bash
   lsof -i :5000
   ```
2. 检查 Docker 日志：
   ```bash
   docker-compose logs registry
   ```

### 认证失败

**问题:** docker login 失败

**解决方案:**
1. 确认用户已添加到系统：
   ```bash
   python3 registry_cli.py user list
   ```
2. 检查 htpasswd 文件：
   ```bash
   cat registry/auth/htpasswd
   ```
3. 重新添加用户：
   ```bash
   python3 registry_cli.py user add <username> <password> --permissions <permissions>
   ```

## 安全建议

### 密码安全

1. **使用强密码**
   - 至少 12 个字符
   - 包含大小写字母、数字和特殊字符
   - 不使用常见单词或容易猜测的模式
   - 建议使用密码生成器生成随机密码

2. **密码管理**
   - 使用密码管理器存储密码
   - 不要在命令历史中保留密码
   - 定期更新密码（建议每 90 天）
   - 不同用户使用不同的密码

### 访问控制

3. **最小权限原则**
   - 只授予用户必要的权限
   - 避免创建过多的管理员账户
   - 定期审查用户权限，移除不需要的权限

4. **用户管理**
   - 及时删除离职人员的账户
   - 为临时访问创建专用账户，使用后及时删除
   - 使用描述性的用户名，便于识别和管理

### 网络安全

5. **HTTPS**
   - 在生产环境中必须使用 HTTPS
   - 使用有效的 SSL/TLS 证书
   - 定期更新证书

6. **防火墙和网络隔离**
   - 限制对 Registry 端口的访问
   - 使用 VPN 或内网访问
   - 配置 IP 白名单（如果可能）

### 数据安全

7. **备份**
   - 定期备份配置文件和镜像数据
   - 验证备份的完整性
   - 将备份存储在安全的位置

8. **审计日志**
   - 启用访问日志记录
   - 定期检查异常访问行为
   - 保留日志至少 90 天

### 容器安全

9. **镜像安全**
   - 只推送经过验证的镜像
   - 定期扫描镜像漏洞
   - 使用镜像签名验证镜像完整性

10. **运行时安全**
    - 使用最小权限运行容器
    - 不在容器中存储敏感信息
    - 定期更新容器和依赖

## 生产环境部署

在生产环境中部署时，建议：

1. **使用 HTTPS**
   - 配置 SSL/TLS 证书
   - 更新 docker-compose.yml 使用 443 端口

2. **使用持久化存储**
   - 将镜像数据存储在可靠的存储系统上
   - 配置备份策略

3. **设置防火墙规则**
   - 限制对 Registry 端口的访问
   - 使用 VPN 或内网访问

4. **监控和日志**
   - 配置日志收集
   - 设置告警规则

## 贡献

欢迎贡献！请提交 Issue 或 Pull Request。

## 许可证

本项目遵循 TeleChat 模型社区许可协议。

## 联系方式

如有问题或建议，请联系：tele_ai@chinatelecom.cn
