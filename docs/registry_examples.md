# TeleChat Docker Registry - 示例配置

## 用户配置示例

这是一个示例用户配置文件 (registry/users.json)，展示如何配置不同权限的用户：

```json
{
  "admin": {
    "username": "admin",
    "password_hash": "htpasswd",
    "permissions": ["admin"],
    "email": "admin@example.com"
  },
  "developer": {
    "username": "developer",
    "password_hash": "htpasswd",
    "permissions": ["view", "pull", "push"],
    "email": "developer@example.com"
  },
  "reader": {
    "username": "reader",
    "password_hash": "htpasswd",
    "permissions": ["view", "pull"],
    "email": "reader@example.com"
  }
}
```

## 镜像配置示例

这是一个示例镜像配置文件 (registry/images.json)，展示如何配置镜像访问权限：

```json
{
  "telechat:7b-fp16": {
    "name": "telechat",
    "tag": "7b-fp16",
    "digest": "sha256:abc123...",
    "created": "2024-12-05T12:00:00Z",
    "size": 14000000000,
    "allowed_users": ["developer", "admin"]
  },
  "telechat:12b-fp16": {
    "name": "telechat",
    "tag": "12b-fp16",
    "digest": "sha256:def456...",
    "created": "2024-12-05T13:00:00Z",
    "size": 24000000000,
    "allowed_users": ["admin"]
  },
  "telechat:7b-int4": {
    "name": "telechat",
    "tag": "7b-int4",
    "digest": "sha256:ghi789...",
    "created": "2024-12-05T14:00:00Z",
    "size": 3500000000,
    "allowed_users": []
  }
}
```

**注意:** 
- `allowed_users` 为空列表 `[]` 表示所有认证用户都可以访问
- `admin` 权限的用户可以访问所有镜像，无需添加到 `allowed_users`

## 环境变量配置

可以通过环境变量自定义配置：

```bash
# 设置仓库地址
export REGISTRY_URL="myregistry.example.com:5000"

# 设置配置目录
export CONFIG_DIR="/etc/telechat/registry"

# 使用自定义配置
python3 registry_cli.py --registry-url $REGISTRY_URL --config-dir $CONFIG_DIR user list
```

## Docker Compose 自定义配置

可以编辑 `docker-compose.yml` 来自定义配置：

### 更改端口

```yaml
services:
  registry:
    ports:
      - "5001:5000"  # 更改为其他端口
```

### 使用 HTTPS

```yaml
services:
  registry:
    environment:
      REGISTRY_HTTP_TLS_CERTIFICATE: /certs/domain.crt
      REGISTRY_HTTP_TLS_KEY: /certs/domain.key
    volumes:
      - ./certs:/certs
```

### 限制存储大小

```yaml
services:
  registry:
    environment:
      REGISTRY_STORAGE_FILESYSTEM_MAXTHREADS: 100
```

## 访问权限策略示例

### 策略 1: 严格控制 (默认拒绝)

每个镜像都必须明确指定允许访问的用户：

```bash
# 注册镜像时指定用户
python3 registry_cli.py image register telechat 7b-fp16 --users developer,tester

# 后续授予其他用户访问权限
python3 registry_cli.py access grant telechat 7b-fp16 new-user
```

### 策略 2: 开放访问 (默认允许)

大部分镜像对所有用户开放，只限制敏感镜像：

```bash
# 公开镜像 (所有用户可访问)
python3 registry_cli.py image register telechat 7b-int4

# 受限镜像 (仅特定用户可访问)
python3 registry_cli.py image register telechat-private 12b-fp16 --users admin,senior-dev
```

### 策略 3: 基于角色的访问控制

为不同角色创建不同权限的用户：

```bash
# 管理员 - 所有权限 (使用强密码)
python3 registry_cli.py user add admin <STRONG_PASSWORD> --permissions admin

# 开发者 - 可以推送和拉取 (使用强密码)
python3 registry_cli.py user add developer <STRONG_PASSWORD> --permissions view,pull,push

# 测试人员 - 只能拉取 (使用强密码)
python3 registry_cli.py user add tester <STRONG_PASSWORD> --permissions view,pull

# 只读用户 - 只能查看 (使用强密码)
python3 registry_cli.py user add viewer <STRONG_PASSWORD> --permissions view
```

## 常见场景

### 场景 1: 团队协作

团队有多个开发人员，需要共享 Docker 镜像：

```bash
# 添加团队成员 (使用强密码)
python3 registry_cli.py user add alice <ALICE_PASSWORD> --permissions view,pull,push
python3 registry_cli.py user add bob <BOB_PASSWORD> --permissions view,pull,push

# 注册共享镜像（所有团队成员可访问）
python3 registry_cli.py image register team-project latest

# 注册私有镜像（只有特定成员可访问）
python3 registry_cli.py image register team-project-private dev --users alice
```

### 场景 2: CI/CD 集成

CI/CD 系统需要自动构建和推送镜像：

```bash
# 创建 CI/CD 服务账号 (使用强令牌)
python3 registry_cli.py user add ci-bot <SECURE_TOKEN> --permissions view,pull,push

# 在 CI/CD 脚本中使用
docker login localhost:5000 -u ci-bot -p $CI_BOT_TOKEN
docker build -t localhost:5000/telechat:latest .
docker push localhost:5000/telechat:latest
```

**注意:** 在生产环境中，应该使用环境变量或密钥管理系统存储凭据，而不是硬编码在脚本中。

### 场景 3: 多环境部署

开发、测试和生产环境使用不同的镜像版本：

```bash
# 开发环境镜像（所有开发人员可访问）
python3 registry_cli.py image register telechat dev

# 测试环境镜像（开发和测试人员可访问）
python3 registry_cli.py image register telechat test --users dev1,dev2,tester1,tester2

# 生产环境镜像（只有运维人员可访问）
python3 registry_cli.py image register telechat prod --users admin,ops1,ops2
```

## 维护操作

### 备份配置

```bash
# 备份配置文件
tar -czf registry-backup-$(date +%Y%m%d).tar.gz registry/

# 恢复配置
tar -xzf registry-backup-20241205.tar.gz
```

### 清理旧镜像

```bash
# 手动清理（需要管理员权限）
docker exec telechat-registry registry garbage-collect /etc/docker/registry/config.yml
```

### 查看日志

```bash
# 查看 Registry 日志
docker compose logs -f registry

# 查看 UI 日志
docker compose logs -f registry-ui
```
