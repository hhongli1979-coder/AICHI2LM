# Docker Registry 快速上手指南

## 5 分钟快速开始

### 步骤 1: 准备环境

确保已安装以下工具：
```bash
docker --version        # Docker 20.10+
docker compose version  # Docker Compose 2.0+
python3 --version       # Python 3.8+
```

如果缺少 apache2-utils：
```bash
# Ubuntu/Debian
sudo apt-get install apache2-utils

# RHEL/CentOS
sudo yum install httpd-tools
```

### 步骤 2: 初始化仓库

运行一键设置脚本：
```bash
chmod +x setup_registry.sh
./setup_registry.sh
```

脚本会：
1. 检查所有依赖
2. 创建必要的目录
3. 生成安全的管理员密码
4. 启动 Docker Registry 和 Web UI

**重要**: 请记录生成的管理员密码！

### 步骤 3: 验证安装

在浏览器中访问：
- **Registry UI**: http://localhost:8080

登录 Registry：
```bash
docker login localhost:5000
# 输入管理员用户名和密码
```

### 步骤 4: 添加第一个用户

```bash
# 添加开发者用户 (使用强密码)
python3 registry_cli.py user add developer <STRONG_PASSWORD> --permissions view,pull,push

# 验证用户列表
python3 registry_cli.py user list
```

### 步骤 5: 构建并推送镜像

使用自动化脚本：
```bash
chmod +x build_image.sh
./build_image.sh 7b-fp16
```

或手动操作：
```bash
# 构建镜像
docker build -t localhost:5000/telechat:7b-fp16 .

# 推送镜像
docker push localhost:5000/telechat:7b-fp16

# 注册镜像到访问控制系统
python3 registry_cli.py image register telechat 7b-fp16
```

## 常用命令速查

### 用户管理

```bash
# 添加用户
python3 registry_cli.py user add <username> <password> --permissions <perms>

# 列出所有用户
python3 registry_cli.py user list

# 更新用户权限
python3 registry_cli.py user update <username> --permissions <perms>

# 删除用户
python3 registry_cli.py user remove <username>
```

### 镜像管理

```bash
# 注册镜像
python3 registry_cli.py image register <name> <tag> [--users user1,user2]

# 列出用户可访问的镜像
python3 registry_cli.py image list-accessible <username>
```

### 访问控制

```bash
# 授予访问权限
python3 registry_cli.py access grant <image> <tag> <username>

# 撤销访问权限
python3 registry_cli.py access revoke <image> <tag> <username>

# 检查访问权限
python3 registry_cli.py access check <username> <image> <tag>
```

### Docker 操作

```bash
# 登录
docker login localhost:5000

# 拉取镜像
docker pull localhost:5000/<image>:<tag>

# 推送镜像
docker push localhost:5000/<image>:<tag>

# 列出镜像
curl -u <username>:<password> http://localhost:5000/v2/_catalog
```

## 权限说明

| 权限 | 说明 | 适用场景 |
|------|------|----------|
| `view` | 查看镜像列表 | 只需要了解有哪些镜像可用 |
| `pull` | 拉取镜像 | 需要使用镜像部署应用 |
| `push` | 推送镜像 | 需要构建和上传新镜像 |
| `delete` | 删除镜像 | 需要清理旧镜像 |
| `admin` | 所有权限 | 系统管理员 |

**推荐权限组合:**
- **开发者**: `view,pull,push`
- **运维人员**: `view,pull,push,delete`
- **测试人员**: `view,pull`
- **只读用户**: `view`

## 访问控制模式

### 模式 1: 公开模式
所有认证用户都可以访问所有镜像：
```bash
python3 registry_cli.py image register <name> <tag>
# 不指定 --users 参数
```

### 模式 2: 限制模式
只有特定用户可以访问：
```bash
python3 registry_cli.py image register <name> <tag> --users user1,user2
```

### 模式 3: 管理员模式
只有管理员可以访问（无需注册）：
```bash
# 只添加管理员用户，不添加镜像的 allowed_users
```

## 常见问题

### Q: 如何重置管理员密码？

```bash
# 删除旧用户
python3 registry_cli.py user remove admin

# 重新添加
python3 registry_cli.py user add admin <NEW_PASSWORD> --permissions admin
```

### Q: 如何查看某个用户的权限？

```bash
python3 registry_cli.py user list | grep <username>
```

### Q: 如何备份配置？

```bash
tar -czf registry-backup.tar.gz registry/
```

### Q: 如何迁移到其他服务器？

1. 备份配置和数据：
```bash
tar -czf registry-data.tar.gz registry/
```

2. 在新服务器上解压：
```bash
tar -xzf registry-data.tar.gz
```

3. 启动服务：
```bash
docker compose up -d
```

### Q: 如何查看 Registry 日志？

```bash
docker compose logs -f registry
```

### Q: 如何停止服务？

```bash
docker compose down
```

### Q: 如何重启服务？

```bash
docker compose restart
```

## 下一步

- 阅读 [完整文档](docker_registry_guide.md)
- 查看 [配置示例](registry_examples.md)
- 了解 [安全最佳实践](docker_registry_guide.md#安全建议)

## 获取帮助

```bash
# CLI 帮助
python3 registry_cli.py --help
python3 registry_cli.py user --help
python3 registry_cli.py image --help
python3 registry_cli.py access --help
```

如有问题，请联系：tele_ai@chinatelecom.cn
