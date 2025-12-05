# TeleChat Docker 安装命令（逐步执行）

## 方式一：自动安装（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/hhongli1979-coder/AICHI2LM.git
cd AICHI2LM

# 2. 运行安装脚本
chmod +x install.sh
./install.sh

# 3. 复制模型文件到 models/7B 目录
# cp -r /your/model/path/* models/7B/

# 完成！访问 http://localhost:8070/docs 和 http://localhost:8501
```

## 方式二：手动安装（逐步执行）

### 第一步：安装 Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
```

### 第二步：安装 GPU 支持

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 第三步：克隆项目

```bash
git clone https://github.com/hhongli1979-coder/AICHI2LM.git
cd AICHI2LM
```

### 第四步：准备模型

```bash
mkdir -p models/7B
# 将你的模型文件复制到 models/7B 目录
# 例如：cp -r /path/to/TeleChat-7B/* models/7B/
```

### 第五步：启动服务

```bash
docker compose up -d
```

### 第六步：查看状态

```bash
docker compose ps
docker compose logs -f
```

## 方式三：使用打包镜像

如果有 telechat-docker-image.tar.gz 文件：

```bash
# 1. 克隆项目
git clone https://github.com/hhongli1979-coder/AICHI2LM.git
cd AICHI2LM

# 2. 加载镜像
gunzip telechat-docker-image.tar.gz
docker load -i telechat-docker-image.tar

# 3. 准备模型
mkdir -p models/7B
# 复制模型文件

# 4. 启动
docker compose up -d
```

## 打包 Docker 镜像

```bash
# 在项目目录执行
chmod +x build_docker_image.sh
./build_docker_image.sh

# 会生成 telechat-docker-image.tar.gz 文件
# 可以传输到其他服务器使用
```

## 常用管理命令

```bash
# 查看日志
docker compose logs -f

# 停止服务
docker compose down

# 重启服务
docker compose restart

# 查看容器状态
docker compose ps

# 进入容器
docker exec -it telechat-service bash
```

## 重要提示

⚠️ **所有 docker compose 命令必须在项目根目录（AICHI2LM）执行！**

如果看到 "no configuration file provided: not found" 错误：

```bash
# 确认当前目录
pwd

# 切换到项目目录
cd /path/to/AICHI2LM

# 确认文件存在
ls docker-compose.yml

# 再执行命令
docker compose up -d
```
