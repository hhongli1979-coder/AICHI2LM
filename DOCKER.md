# TeleChat Docker 部署指南

## 概述

本指南提供使用 Docker 部署 TeleChat 模型服务的方法，支持快速启动和环境隔离。

## 前置要求

1. **Docker**: 版本 20.10 或更高
2. **Docker Compose**: 版本 2.0 或更高
3. **NVIDIA Docker Runtime** (用于 GPU 支持):
   ```bash
   # 安装 NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

4. **模型文件**: 确保模型文件已下载到 `models/` 目录

## 快速开始

### 方式一：使用 Docker Compose (推荐)

1. **启动服务**:
   ```bash
   docker-compose up -d
   ```

2. **查看日志**:
   ```bash
   docker-compose logs -f
   ```

3. **访问服务**:
   - API 文档: http://localhost:8070/docs
   - Web 界面: http://localhost:8501

4. **停止服务**:
   ```bash
   docker-compose down
   ```

### 方式二：使用 Docker 命令

1. **构建镜像**:
   ```bash
   docker build -t telechat:latest .
   ```

2. **运行容器**:
   ```bash
   docker run -d \
     --name telechat-service \
     --gpus all \
     -p 8070:8070 \
     -p 8501:8501 \
     -v $(pwd)/models:/app/models \
     -e CUDA_VISIBLE_DEVICES=0 \
     telechat:latest
   ```

3. **查看日志**:
   ```bash
   docker logs -f telechat-service
   ```

4. **停止容器**:
   ```bash
   docker stop telechat-service
   docker rm telechat-service
   ```

## 配置选项

### 环境变量

可以通过环境变量自定义配置：

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `MODEL_PATH` | `/app/models/7B` | 模型文件路径 |
| `CUDA_VISIBLE_DEVICES` | `0` | 使用的 GPU 设备 |
| `API_PORT` | `8070` | API 服务端口 |
| `WEB_PORT` | `8501` | Web 服务端口 |

### 使用 .env 文件

创建 `.env` 文件：

```bash
MODEL_PATH=/app/models/12B
CUDA_VISIBLE_DEVICES=0,1
API_PORT=8080
WEB_PORT=8502
```

然后启动：

```bash
docker-compose --env-file .env up -d
```

## 高级用法

### 多 GPU 部署

编辑 `docker-compose.yml`，修改环境变量：

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 使用不同的模型

```bash
docker-compose up -d -e MODEL_PATH=/app/models/12B
```

### 开发模式

使用卷挂载实现代码热重载：

```bash
docker-compose up
```

代码变更会自动同步到容器中。

### CPU 模式

如果没有 GPU，修改 `docker-compose.yml`：

1. 移除 `deploy.resources` 部分
2. 设置 `CUDA_VISIBLE_DEVICES=""`

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=
```

## 故障排除

### 1. GPU 不可用

检查 NVIDIA Docker 运行时：

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
```

### 2. 端口冲突

修改 `docker-compose.yml` 中的端口映射：

```yaml
ports:
  - "8080:8070"  # 使用主机 8080 端口
  - "8502:8501"  # 使用主机 8502 端口
```

### 3. 模型文件未找到

确保模型文件在正确的路径：

```bash
ls -la ./models/7B
```

### 4. 内存不足

增加 Docker 内存限制（在 Docker Desktop 设置中）或使用量化模型。

### 5. 构建失败

清理并重新构建：

```bash
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

## 性能优化

### 1. 使用预构建镜像

可以将构建好的镜像推送到私有仓库：

```bash
docker tag telechat:latest your-registry/telechat:latest
docker push your-registry/telechat:latest
```

### 2. 优化镜像大小

使用多阶段构建（已在 Dockerfile 中实现）。

### 3. 缓存模型

首次运行后，模型会被缓存在卷中，后续启动会更快。

## 生产环境部署

### 使用 Docker Swarm

```bash
docker stack deploy -c docker-compose.yml telechat
```

### 使用 Kubernetes

参考 `k8s/` 目录中的配置文件（如果有）。

## 安全建议

1. **不要在镜像中包含模型文件** - 使用卷挂载
2. **限制容器权限** - 使用非 root 用户运行
3. **使用私有仓库** - 存储自定义镜像
4. **定期更新基础镜像** - 修复安全漏洞

## 监控和日志

### 查看实时日志

```bash
docker-compose logs -f telechat
```

### 导出日志

```bash
docker-compose logs telechat > telechat.log
```

### 监控资源使用

```bash
docker stats telechat-service
```

## 参考资源

- [Docker 官方文档](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose 文档](https://docs.docker.com/compose/)

## 贡献

欢迎提交问题和改进建议！

## 许可证

遵循 TeleChat 项目的许可证。
