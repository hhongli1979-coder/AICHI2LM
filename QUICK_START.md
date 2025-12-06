# TeleChat Docker 快速部署（5分钟上手）

> 📖 **在 GitHub 上查看此文档**：https://github.com/hhongli1979-coder/AICHI2LM/blob/main/QUICK_START.md

## 🚀 三步完成部署

### 步骤 1：安装 Docker 和 GPU 支持

**复制以下命令到终端，逐行执行：**

```bash
# 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# 安装 NVIDIA GPU 支持
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 步骤 2：下载项目

```bash
# 进入你想要安装的目录（例如 /www/wwwroot）
cd /www/wwwroot

# 克隆项目
git clone https://github.com/hhongli1979-coder/AICHI2LM.git

# 进入项目目录
cd AICHI2LM
```

### 步骤 3：准备模型并启动

```bash
# 创建模型目录
mkdir -p models/7B

# 把你的模型文件复制到 models/7B 目录
# 例如：cp -r /path/to/your/model/* models/7B/

# 启动服务
docker compose up -d

# 查看启动日志
docker compose logs -f
```

## ✅ 访问服务

部署完成后，在浏览器访问：

- **API 文档**: http://你的服务器IP:8070/docs
- **Web 界面**: http://你的服务器IP:8501

如果在本地：
- API: http://localhost:8070/docs
- Web: http://localhost:8501

## 📋 常用命令

```bash
# 必须在 AICHI2LM 目录下执行以下命令
cd /www/wwwroot/AICHI2LM

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f

# 停止服务
docker compose down

# 重启服务
docker compose restart

# 更新代码后重启
git pull
docker compose restart
```

## ❓ 遇到问题？

### 问题 1：提示 "no configuration file provided"

**原因**：不在项目目录

**解决**：
```bash
# 确认当前位置
pwd

# 进入项目目录
cd /www/wwwroot/AICHI2LM

# 确认文件存在
ls docker-compose.yml

# 再执行命令
docker compose up -d
```

### 问题 2：找不到 AICHI2LM 目录

**原因**：还没克隆项目

**解决**：
```bash
cd /www/wwwroot
git clone https://github.com/hhongli1979-coder/AICHI2LM.git
cd AICHI2LM
docker compose up -d
```

### 问题 3：端口被占用

**解决**：修改端口
```bash
# 使用不同的端口
API_PORT=8080 WEB_PORT=8502 docker compose up -d
```

### 问题 4：GPU 不可用

**验证 GPU**：
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
```

如果失败，重新安装 nvidia-container-toolkit（执行步骤 1 的 GPU 支持部分）

### 问题 5：模型加载失败

**检查模型文件**：
```bash
# 确认模型文件存在
ls -la models/7B/

# 应该看到 .bin, .json, tokenizer 等文件
```

## 📞 获取帮助

查看更多文档：
- 详细文档：查看项目中的 `DOCKER.md`
- 命令列表：查看项目中的 `命令.txt`
- 安装步骤：查看项目中的 `INSTALL_STEPS.md`

## 🎯 完整流程示例

从零开始的完整命令（适合直接复制粘贴）：

```bash
# 1. 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh && sudo usermod -aG docker $USER && newgrp docker

# 2. 安装 GPU 支持
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker

# 3. 克隆并启动
cd /www/wwwroot && git clone https://github.com/hhongli1979-coder/AICHI2LM.git && cd AICHI2LM && mkdir -p models/7B && echo "请把模型文件复制到 $(pwd)/models/7B 目录，然后执行: docker compose up -d"
```

复制模型后：
```bash
cd /www/wwwroot/AICHI2LM
docker compose up -d
```

---

**注意**：所有 `docker compose` 命令必须在 `AICHI2LM` 项目目录下执行！
