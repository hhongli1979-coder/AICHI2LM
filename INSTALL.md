# TeleChat Docker 快速安装指南

## ⚠️ 重要提示

**必须在项目根目录（包含 docker-compose.yml 的目录）执行 Docker 命令！**

如果你看到 "no configuration file provided: not found" 错误，说明你不在正确的目录。

## 🚀 快速安装（5分钟完成）

### 1. 克隆项目

```bash
git clone https://github.com/hhongli1979-coder/AICHI2LM.git
cd AICHI2LM  # ← 重要：进入项目目录
```

### 2. 准备模型

```bash
# 创建模型目录
mkdir -p models/7B

# 下载或复制你的模型文件到 models/7B 目录
# 例如：
# cp -r /path/to/your/TeleChat-7B/* models/7B/
```

### 3. 启动服务

```bash
# 方式一：使用预构建镜像（如果有提供）
docker-compose up -d

# 方式二：自己构建镜像
./build_docker_image.sh  # 构建并打包镜像
docker-compose up -d     # 启动服务
```

### 4. 访问服务

- **API 文档**: http://localhost:8070/docs
- **Web 界面**: http://localhost:8501

## 📦 离线安装（使用打包镜像）

如果你有 `telechat-docker-image.tar.gz` 文件：

```bash
# 1. 克隆项目
git clone https://github.com/hhongli1979-coder/AICHI2LM.git
cd AICHI2LM

# 2. 加载镜像
gunzip telechat-docker-image.tar.gz
docker load -i telechat-docker-image.tar

# 3. 准备模型（同上）
mkdir -p models/7B
# 复制模型文件...

# 4. 启动服务
docker-compose up -d
```

## 🔧 前置条件

### 安装 Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
```

### 安装 GPU 支持（可选）

```bash
# 安装 NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## ❓ 常见问题

### "no configuration file provided: not found"

**原因**: 你不在项目根目录

**解决**:
```bash
cd /path/to/AICHI2LM  # 进入项目目录
pwd                    # 确认当前目录
ls docker-compose.yml  # 确认文件存在
docker-compose up -d   # 再次启动
```

### 端口被占用

```bash
# 修改端口
API_PORT=8080 WEB_PORT=8502 docker-compose up -d
```

### GPU 不可用

```bash
# 验证 GPU 支持
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi

# 如果失败，重新安装 nvidia-container-toolkit
```

### 找不到模型

```bash
# 确认模型目录结构
ls -la models/7B

# 应该包含模型文件（.bin, .json, tokenizer 等）
```

## 📝 完整文档

- 详细指南: [DOCKER.md](./DOCKER.md)
- 快速参考: [DOCKER_QUICKREF.md](./DOCKER_QUICKREF.md)
- 部署说明: [DEPLOYMENT.md](./DEPLOYMENT.md)

## 🆘 需要帮助？

1. 查看日志: `docker-compose logs -f`
2. 检查容器状态: `docker-compose ps`
3. 查看完整文档: `DOCKER.md`
