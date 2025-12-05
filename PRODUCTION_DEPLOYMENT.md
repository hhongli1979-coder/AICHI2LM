# TeleChat 生产部署指南

## 目录

- [概述](#概述)
- [系统要求](#系统要求)
- [部署方式](#部署方式)
  - [Docker Compose 部署（推荐）](#docker-compose-部署推荐)
  - [Docker 独立部署](#docker-独立部署)
  - [Kubernetes 部署](#kubernetes-部署)
- [配置说明](#配置说明)
- [监控和日志](#监控和日志)
- [性能优化](#性能优化)
- [故障排查](#故障排查)
- [安全建议](#安全建议)

## 概述

本指南详细介绍如何将 TeleChat 模型部署到生产环境。生产部署包括：

- **容器化部署**：使用 Docker 容器确保环境一致性
- **反向代理**：使用 Nginx 提供负载均衡和 SSL 终止
- **健康检查**：自动监控服务状态
- **日志管理**：集中化日志收集和分析
- **高可用性**：支持多实例部署和自动重启

## 系统要求

### 硬件要求

| 组件 | 最小配置 | 推荐配置 |
|------|---------|---------|
| CPU | 8 核 | 16 核+ |
| 内存 | 32 GB | 64 GB+ |
| GPU | NVIDIA GPU 16GB+ (如 V100, A10) | NVIDIA A100 40GB+ |
| 存储 | 100 GB SSD | 500 GB+ NVMe SSD |
| 网络 | 1 Gbps | 10 Gbps |

### 软件要求

- **操作系统**：Ubuntu 20.04+ / CentOS 7+ / RHEL 8+
- **Docker**：20.10+
- **Docker Compose**：2.0+
- **NVIDIA Container Toolkit**：用于 GPU 支持
- **CUDA**：11.8+
- **Python**：3.8+（用于手动部署）

### GPU 驱动安装

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y nvidia-driver-530 nvidia-utils-530

# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 验证安装
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 部署方式

### Docker Compose 部署（推荐）

Docker Compose 是最简单的生产部署方式，自动管理多个容器和网络配置。

#### 1. 准备模型文件

```bash
# 下载或复制模型文件到 models 目录
mkdir -p models/7B
# 将模型文件放入 models/7B/ 目录
# 目录结构应该是:
# models/7B/
#   ├── config.json
#   ├── generation_config.json
#   ├── pytorch_model.bin (或 *.safetensors)
#   ├── tokenizer_config.json
#   └── ...
```

#### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.production .env

# 编辑 .env 文件，根据需要修改配置
vim .env
```

#### 3. 构建 Docker 镜像

```bash
# 构建镜像
docker-compose build

# 或者使用缓存加速
docker-compose build --parallel
```

#### 4. 启动服务

```bash
# 启动所有服务（后台运行）
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看服务日志
docker-compose logs -f telechat
docker-compose logs -f nginx
```

#### 5. 验证部署

```bash
# 检查健康状态
curl http://localhost/health

# 访问 API 文档
# 浏览器打开: http://your-server-ip/api/docs

# 访问 Web 界面
# 浏览器打开: http://your-server-ip/
```

#### 6. 停止服务

```bash
# 停止服务
docker-compose down

# 停止服务并删除卷（注意：会删除日志）
docker-compose down -v
```

### Docker 独立部署

如果不使用 Docker Compose，可以手动运行 Docker 容器。

#### 1. 构建镜像

```bash
docker build -t telechat:latest .
```

#### 2. 运行容器

```bash
# 运行 TeleChat 服务
docker run -d \
  --name telechat-service \
  --gpus all \
  -p 8070:8070 \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/logs:/app/logs \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e MODEL_PATH=/app/models/7B \
  --restart unless-stopped \
  telechat:latest

# 运行 Nginx 反向代理
docker run -d \
  --name telechat-nginx \
  -p 80:80 \
  -p 443:443 \
  -v $(pwd)/nginx/nginx.conf:/etc/nginx/nginx.conf:ro \
  -v $(pwd)/nginx/conf.d:/etc/nginx/conf.d:ro \
  -v $(pwd)/nginx/ssl:/etc/nginx/ssl:ro \
  -v $(pwd)/nginx/logs:/var/log/nginx \
  --link telechat-service:telechat \
  --restart unless-stopped \
  nginx:alpine
```

#### 3. 查看日志

```bash
# 查看 TeleChat 日志
docker logs -f telechat-service

# 查看 Nginx 日志
docker logs -f telechat-nginx
```

### Kubernetes 部署

对于大规模生产环境，建议使用 Kubernetes 进行编排。

#### 1. 创建 Kubernetes 配置文件

```yaml
# telechat-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: telechat-deployment
  labels:
    app: telechat
spec:
  replicas: 2
  selector:
    matchLabels:
      app: telechat
  template:
    metadata:
      labels:
        app: telechat
    spec:
      containers:
      - name: telechat
        image: telechat:latest
        ports:
        - containerPort: 8070
        - containerPort: 8501
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: MODEL_PATH
          value: "/app/models/7B"
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
        - name: logs
          mountPath: /app/logs
        livenessProbe:
          httpGet:
            path: /health
            port: 8070
          initialDelaySeconds: 120
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8070
          initialDelaySeconds: 60
          periodSeconds: 10
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: telechat-models-pvc
      - name: logs
        persistentVolumeClaim:
          claimName: telechat-logs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: telechat-service
spec:
  selector:
    app: telechat
  ports:
  - name: api
    port: 8070
    targetPort: 8070
  - name: web
    port: 8501
    targetPort: 8501
  type: LoadBalancer
```

#### 2. 部署到 Kubernetes

```bash
# 应用配置
kubectl apply -f telechat-deployment.yaml

# 查看部署状态
kubectl get deployments
kubectl get pods
kubectl get services

# 查看日志
kubectl logs -f deployment/telechat-deployment
```

## 配置说明

### 环境变量配置

在 `.env` 或 `.env.production` 文件中配置：

| 变量名 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `MODEL_PATH` | 模型文件路径 | `/app/models/7B` | `/app/models/12B` |
| `CUDA_VISIBLE_DEVICES` | GPU 设备编号 | `0` | `0,1` |
| `API_PORT` | API 服务端口 | `8070` | `8080` |
| `WEB_PORT` | Web 服务端口 | `8501` | `8502` |
| `LOG_LEVEL` | 日志级别 | `INFO` | `DEBUG` |

### 部署配置文件

编辑 `deploy_config.prod.yaml`：

```yaml
# 模型路径
model_path: '/app/models/7B'

# 服务配置
api_host: '0.0.0.0'
api_port: 8070
web_host: '0.0.0.0'
web_port: 8501

# GPU 配置
gpu_devices: '0'

# 性能配置
max_workers: 4
timeout: 600
```

### Nginx 配置

#### SSL/TLS 配置

要启用 HTTPS，需要准备 SSL 证书：

1. **获取证书**：
   - 使用 Let's Encrypt（免费）
   - 购买商业证书
   - 使用自签名证书（仅用于测试）

2. **配置证书**：

```bash
# 将证书放入 nginx/ssl 目录
cp cert.pem nginx/ssl/
cp key.pem nginx/ssl/

# 修改 nginx/conf.d/telechat.conf，取消 HTTPS 配置的注释
vim nginx/conf.d/telechat.conf
```

3. **重启 Nginx**：

```bash
docker-compose restart nginx
```

## 监控和日志

### 健康检查

服务提供了健康检查端点：

```bash
# 检查服务健康状态
curl http://localhost/health

# 响应示例
{
  "status": "healthy",
  "service": "TeleChat API",
  "model_loaded": true,
  "gpu_info": {
    "gpu_available": true,
    "gpu_count": 1,
    "current_device": 0
  }
}
```

### 日志管理

日志文件位置：

- **TeleChat 服务日志**：`./logs/` 目录
- **Nginx 访问日志**：`./nginx/logs/access.log`
- **Nginx 错误日志**：`./nginx/logs/error.log`

查看日志：

```bash
# 实时查看 TeleChat 日志
docker-compose logs -f telechat

# 查看 Nginx 日志
docker-compose logs -f nginx

# 查看最近 100 行日志
docker-compose logs --tail=100 telechat
```

### 日志轮转

配置日志轮转以防止日志文件过大：

```bash
# /etc/logrotate.d/telechat
/path/to/AICHI2LM/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
}

/path/to/AICHI2LM/nginx/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
    sharedscripts
    postrotate
        docker-compose -f /path/to/AICHI2LM/docker-compose.yml exec nginx nginx -s reload
    endscript
}
```

### 监控集成

#### Prometheus 监控

可以集成 Prometheus 进行指标收集：

```yaml
# 在 docker-compose.yml 中添加
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
```

#### Grafana 可视化

```yaml
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
```

## 性能优化

### GPU 优化

1. **多 GPU 配置**：

```bash
# 在 .env 中配置
CUDA_VISIBLE_DEVICES=0,1,2,3
```

2. **混合精度**：

模型默认使用 `torch.float16` 以提高性能。

3. **批处理**：

对于高并发场景，考虑实现请求批处理。

### 内存优化

```yaml
# 在 docker-compose.yml 中限制内存
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          memory: 16G
```

### 网络优化

```nginx
# 在 nginx.conf 中优化
worker_processes auto;
worker_connections 2048;
keepalive_timeout 65;
keepalive_requests 100;
```

## 故障排查

### 常见问题

#### 1. 容器启动失败

```bash
# 查看容器日志
docker-compose logs telechat

# 常见原因：
# - 模型文件缺失或路径错误
# - GPU 不可用
# - 端口被占用
# - 内存不足
```

**解决方法**：

```bash
# 检查模型路径
ls -lh models/7B/

# 检查 GPU
nvidia-smi

# 检查端口
netstat -tlnp | grep -E '8070|8501'

# 检查内存
free -h
```

#### 2. 健康检查失败

```bash
# 检查健康端点
curl -v http://localhost:8070/health

# 查看详细日志
docker-compose logs telechat | grep -i error
```

#### 3. 推理速度慢

**可能原因**：
- GPU 未正确使用
- 模型未使用混合精度
- 并发请求过多

**解决方法**：

```bash
# 检查 GPU 使用情况
nvidia-smi

# 增加 GPU 资源
CUDA_VISIBLE_DEVICES=0,1

# 调整并发数
MAX_WORKERS=4
```

#### 4. 内存溢出（OOM）

```bash
# 减少最大输入长度
max_length: 2048

# 使用量化模型
model_path: '/app/models/7B-int4'

# 增加交换空间（临时方案）
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 调试模式

启用调试模式获取更详细的日志：

```bash
# 修改 .env
LOG_LEVEL=DEBUG

# 重启服务
docker-compose restart telechat
```

## 安全建议

### 1. 网络安全

- **使用 HTTPS**：在生产环境中始终使用 SSL/TLS
- **防火墙配置**：仅开放必要的端口
- **使用私有网络**：将服务部署在私有网络中

```bash
# 配置防火墙（UFW 示例）
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 2. 访问控制

在 Nginx 配置中添加认证：

```nginx
# 在 location 块中添加
auth_basic "Restricted Access";
auth_basic_user_file /etc/nginx/.htpasswd;
```

生成密码文件：

```bash
# 安装 htpasswd
sudo apt-get install apache2-utils

# 创建用户
sudo htpasswd -c /path/to/nginx/.htpasswd username
```

### 3. API 限流

在 Nginx 中配置限流：

```nginx
http {
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
        # ... 其他配置
    }
}
```

### 4. 定期更新

```bash
# 更新系统包
sudo apt-get update && sudo apt-get upgrade -y

# 更新 Docker 镜像
docker-compose pull
docker-compose up -d
```

### 5. 数据安全

- **加密敏感数据**：不要在镜像中包含敏感信息
- **使用 secrets**：通过 Docker secrets 或 Kubernetes secrets 管理密钥
- **定期备份**：备份模型文件和配置

```bash
# 备份脚本示例
#!/bin/bash
BACKUP_DIR="/backup/telechat-$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR
cp -r models/ $BACKUP_DIR/
cp -r nginx/conf.d/ $BACKUP_DIR/
cp .env $BACKUP_DIR/
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR
```

## 生产环境检查清单

部署前请确认以下事项：

- [ ] 系统要求满足（CPU、内存、GPU、存储）
- [ ] Docker 和 NVIDIA Container Toolkit 已安装
- [ ] 模型文件已下载并放置在正确位置
- [ ] 环境变量已正确配置
- [ ] 防火墙规则已设置
- [ ] SSL 证书已配置（如需 HTTPS）
- [ ] 健康检查端点正常工作
- [ ] 日志轮转已配置
- [ ] 监控系统已设置
- [ ] 备份策略已制定
- [ ] 负载测试已完成
- [ ] 安全审计已通过
- [ ] 文档已更新

## 扩展阅读

- [Docker 官方文档](https://docs.docker.com/)
- [Nginx 官方文档](https://nginx.org/en/docs/)
- [Kubernetes 官方文档](https://kubernetes.io/docs/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [TeleChat 模型文档](./DEPLOYMENT.md)

## 技术支持

如有问题，请：

1. 查看[故障排查](#故障排查)部分
2. 查看项目 Issues
3. 提交新的 Issue 并附上：
   - 系统信息（OS、Docker 版本等）
   - 错误日志
   - 复现步骤

---

**版本**: 1.0.0  
**更新日期**: 2024-12-05  
**维护者**: TeleChat Team
