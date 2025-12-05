# TeleChat 一键部署工具使用指南

## 概述

TeleChat 一键部署工具提供了简单快捷的方式来部署 TeleChat 模型的 API 和 Web 服务。无需手动启动多个服务，一个命令即可完成部署。

## 功能特性

- ✅ 自动启动 API 和 Web 服务
- ✅ 智能检查依赖项和模型文件
- ✅ 支持自定义配置（端口、GPU、模型路径等）
- ✅ 跨平台支持（Linux、macOS、Windows）
- ✅ 优雅的服务停止和清理
- ✅ 详细的日志输出

## 快速开始

### 1. 准备工作

确保已安装必要的依赖：

```bash
pip install -r requirements.txt
```

确保模型文件已下载到正确路径（默认：`../models/7B`）

### 2. 一键部署

#### 方式一：Python 脚本（推荐）

跨平台，功能最完整：

```bash
python deploy.py
```

#### 方式二：Shell 脚本

适用于 Linux 和 macOS：

```bash
chmod +x deploy.sh  # 首次运行需要添加执行权限
./deploy.sh
```

#### 方式三：批处理脚本

适用于 Windows：

```cmd
deploy.bat
```

### 3. 访问服务

部署成功后，可以通过以下地址访问：

- **API 文档**: http://localhost:8070/docs
- **Web 界面**: http://localhost:8501

### 4. 停止服务

按 `Ctrl+C` 即可优雅地停止所有服务。

## 配置选项

### 方式一：使用配置文件

创建或修改 `deploy_config.yaml` 文件：

```yaml
# 模型路径
model_path: '../models/7B'

# API 服务配置
api_host: '0.0.0.0'
api_port: 8070

# Web 服务配置
web_host: '0.0.0.0'
web_port: 8501

# GPU 设备配置
gpu_devices: '0'

# 服务启动检查间隔（秒）
check_interval: 2

# 最大等待时间（秒）
max_wait_time: 60
```

然后使用配置文件启动：

```bash
python deploy.py --config deploy_config.yaml
```

### 方式二：使用命令行参数

#### Python 脚本

```bash
# 指定模型路径
python deploy.py --model-path /path/to/model

# 指定 GPU 设备
python deploy.py --gpu 0,1

# 指定端口
python deploy.py --api-port 8080 --web-port 8502

# 组合使用
python deploy.py --model-path ../models/12B --gpu 0 --api-port 8080
```

查看所有选项：

```bash
python deploy.py --help
```

#### Shell 脚本

```bash
# 指定模型路径
./deploy.sh --model ../models/12B

# 指定 GPU 设备
./deploy.sh --gpu 0,1

# 指定端口
./deploy.sh --api-port 8080 --web-port 8502

# 组合使用
./deploy.sh --model ../models/12B --gpu 0,1 --api-port 8080
```

查看所有选项：

```bash
./deploy.sh --help
```

### 方式三：使用环境变量

#### Linux/macOS

```bash
export MODEL_PATH="../models/12B"
export CUDA_VISIBLE_DEVICES="0,1"
export API_PORT="8080"
export WEB_PORT="8502"

./deploy.sh
```

#### Windows

```cmd
set MODEL_PATH=..\models\12B
set CUDA_VISIBLE_DEVICES=0,1
set API_PORT=8080
set WEB_PORT=8502

deploy.bat
```

## 常见问题

### 1. 端口已被占用

如果提示端口已被占用，可以：

- 停止占用端口的进程
- 使用不同的端口：`python deploy.py --api-port 8080 --web-port 8502`

### 2. 找不到模型文件

确保：

- 模型路径正确
- 模型文件已完整下载
- 使用 `--model-path` 参数指定正确的路径

### 3. GPU 不可用

- 确保已安装 CUDA 和 PyTorch GPU 版本
- 检查 GPU 设备编号是否正确
- 如需使用 CPU，可以设置 `CUDA_VISIBLE_DEVICES=""`

### 4. 依赖包缺失

运行以下命令安装所有依赖：

```bash
pip install -r requirements.txt
```

### 5. 服务启动失败

查看日志文件以获取详细错误信息：

- **Python 脚本**: 查看控制台输出
- **Shell 脚本**: 查看 `/tmp/telechat_api.log` 和 `/tmp/telechat_web.log`
- **Windows 批处理**: 查看 `%TEMP%\telechat_api.log` 和 `%TEMP%\telechat_web.log`

## 高级用法

### 多 GPU 部署

```bash
python deploy.py --gpu 0,1,2,3
```

### 使用量化模型

```bash
python deploy.py --model-path ../models/7B-int4
```

### 自定义启动超时时间

修改 `deploy_config.yaml`：

```yaml
max_wait_time: 120  # 增加到 120 秒
```

### 仅启动 API 服务

如果只需要 API 服务，可以修改 `deploy.py` 或手动运行：

```bash
cd service
python telechat_service.py
```

## 技术细节

### 服务架构

```
┌─────────────────┐
│   deploy.py     │  主部署脚本
└────────┬────────┘
         │
         ├─────────────────────┐
         ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│  API Service    │   │  Web Service    │
│  (FastAPI)      │◄──│  (Streamlit)    │
│  Port: 8070     │   │  Port: 8501     │
└─────────────────┘   └─────────────────┘
         │
         ▼
┌─────────────────┐
│  TeleChat Model │
│  (PyTorch)      │
└─────────────────┘
```

### 启动流程

1. 检查 Python 版本和依赖包
2. 验证模型路径和文件
3. 检查端口可用性
4. 设置 GPU 环境变量
5. 启动 API 服务（后台进程）
6. 等待 API 服务就绪（健康检查）
7. 启动 Web 服务（后台进程）
8. 监控服务运行状态
9. 捕获停止信号，优雅关闭服务

### 进程管理

- **Python 脚本**: 使用 `subprocess` 管理子进程，支持优雅关闭
- **Shell 脚本**: 使用 PID 文件跟踪进程，支持信号处理
- **Windows 批处理**: 使用 `start /b` 启动后台进程

## 贡献

欢迎提交问题和改进建议！

## 许可证

遵循 TeleChat 项目的许可证。
