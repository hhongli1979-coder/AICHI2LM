# 快速开始

本教程旨在帮助使用者快速进行Telechat模型的部署开发，主要包括：

## 一键本地部署 🚀

TeleChat现在支持一键本地部署，无需手动启动多个服务。

### 方式一：使用Python脚本（推荐，跨平台）

```bash
# 使用默认配置
python deploy.py

# 使用自定义配置文件
python deploy.py --config deploy_config.yaml

# 指定模型路径和GPU
python deploy.py --model-path /path/to/model --gpu 0

# 指定端口
python deploy.py --api-port 8080 --web-port 8502
```

### 方式二：使用Shell脚本（Linux/Mac）

```bash
# 使用默认配置
./deploy.sh

# 指定模型路径
./deploy.sh --model ../models/12B

# 指定GPU设备
./deploy.sh --gpu 0,1

# 指定端口
./deploy.sh --api-port 8080 --web-port 8502
```

### 方式三：使用批处理脚本（Windows）

```cmd
REM 使用默认配置
deploy.bat

REM 设置环境变量后运行
set MODEL_PATH=..\models\12B
set CUDA_VISIBLE_DEVICES=0
deploy.bat
```

### 配置说明

可以通过修改 `deploy_config.yaml` 文件来配置部署参数：

```yaml
# 模型路径
model_path: '../models/7B'

# API服务配置
api_host: '0.0.0.0'
api_port: 8070

# Web服务配置
web_host: '0.0.0.0'
web_port: 8501

# GPU设备配置
gpu_devices: '0'
```

部署成功后，可以通过以下地址访问：
- **API文档**: http://localhost:8070/docs
- **Web界面**: http://localhost:8501

按 `Ctrl+C` 停止所有服务。

---

## 环境准备与依赖安装

### 系统要求

在开始之前，请确保您的系统满足以下要求：

**硬件要求**:
- GPU: NVIDIA GPU with CUDA support (推荐8GB+显存用于推理，40GB+用于训练)
- 内存: 32GB+ RAM
- 存储: 50GB+可用磁盘空间

**软件要求**:
- 操作系统: Linux (Ubuntu 18.04+推荐) / macOS / Windows 10/11
- Python: 3.8, 3.9, 或 3.10
- CUDA: 11.6+ (如果使用GPU)
- Git & Git LFS

### 本地环境安装

#### 1. 创建Python虚拟环境

强烈建议使用虚拟环境来隔离依赖：

```bash
# 方式1: 使用conda (推荐)
conda create -n telechat python=3.9
conda activate telechat

# 方式2: 使用venv
python3 -m venv telechat_env
source telechat_env/bin/activate  # Linux/Mac
# telechat_env\Scripts\activate  # Windows
```

#### 2. 安装PyTorch

根据您的CUDA版本选择合适的PyTorch版本：

```bash
# 查看CUDA版本
nvcc --version

# CUDA 11.7
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# CUDA 11.8  
pip install torch==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# CPU版本 (仅用于推理测试，不推荐训练)
pip install torch==1.13.1
```

详细版本对照请参考: [PyTorch官网](https://pytorch.org/get-started/locally/)

#### 3. 克隆代码仓库

```bash
git clone https://github.com/Tele-AI/TeleChat.git
cd TeleChat
```

#### 4. 安装项目依赖

```bash
# 安装所有依赖
pip install -r requirements.txt
```

主要依赖包括：
- `transformers==4.30.0`: Hugging Face模型库
- `deepspeed==0.8.3`: 分布式训练框架
- `accelerate>=0.24.1`: 训练加速库
- `fastapi>=0.109.1`: API服务框架
- `streamlit>=1.30.0`: Web界面框架
- `auto-gptq==0.3.0`: 量化工具

#### 5. 安装FlashAttention2 (可选但推荐)

FlashAttention2可以显著提升训练和推理速度：

```bash
pip install flash-attn --no-build-isolation
```

**注意事项**:
- FlashAttention2需要编译，可能需要10-30分钟
- 需要安装CUDA开发工具包
- 如果安装失败，可以跳过，模型会自动降级到标准注意力机制

#### 6. 验证安装

```bash
# 验证PyTorch和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 验证transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 验证deepspeed
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
```

预期输出示例：
```
PyTorch: 1.13.1+cu117
CUDA: True
Transformers: 4.30.0
DeepSpeed: 0.8.3
```

### 常见安装问题

#### 问题1: CUDA版本不匹配

**错误信息**: `RuntimeError: CUDA error: no kernel image is available`

**解决方案**:
```bash
# 检查CUDA版本
nvcc --version

# 重新安装匹配的PyTorch版本
pip uninstall torch
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

#### 问题2: FlashAttention2编译失败

**错误信息**: 编译过程中出现各种错误

**解决方案**:
- 确保安装了CUDA开发工具: `sudo apt-get install cuda-toolkit`
- 尝试使用预编译wheel: 在[这里](https://github.com/Dao-AILab/flash-attention/releases)下载
- 如果仍然失败，可以跳过FlashAttention2，使用标准注意力机制

#### 问题3: 显存不足

**错误信息**: `CUDA out of memory`

**解决方案**:
- 使用量化模型 (int8/int4)
- 降低batch size
- 使用gradient checkpointing
- 启用CPU offload

#### 问题4: Windows系统安装问题

**常见问题**: 某些包无法编译

**解决方案**:
```bash
# 1. 安装Microsoft C++ Build Tools
# 下载: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 2. 使用Anaconda替代pip
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 3. 考虑使用WSL2
wsl --install
```

---

## 资源获取


### 镜像下载

为了便于大家快速上手，我们提供了可运行的环境镜像，下载地址：[镜像下载](https://cloud.189.cn/t/EbAriaQfa2mm) （访问码：2uik）

### 模型权重下载


| 模型版本  | 下载链接           |
|---------| ----------------- |
| 7B-FP16 | [TeleChat-FP16](https://huggingface.co/Tele-AI/Telechat-7B) |
| 7B-int8 | [TeleChat-int8](https://huggingface.co/Tele-AI/Telechat-7B-int8) |
| 7B-int4 | [TeleChat-int4](https://huggingface.co/Tele-AI/Telechat-7B-int4) |

## 环境配置


### 镜像开发
**教程中1.2版本为例，后续请修改版本号。**

获取镜像，并完成解压得到tar文件。

![镜像tar包](../images/镜像tar包.png)

导入镜像包

```shell
sudo docker load -i telechat-public_1.2.tar
```

![镜像导入](../images/镜像导入过程.png)

启动容器，其中NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7代表挂载编号0-7的8张GPU显卡，请自行修改

```shell
sudo docker run -itd  --name telechat --runtime=nvidia  --shm-size=256g -e NVIDIA_DRIVER_CAPABILITIES=compute,utility  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 telechat-public:1.2 bash
```

![容器启动](../images/容器启动.png)

复制代码和镜像到容器内

- 例如 复制本地模型文件夹 telechat_opensource_7B_huggingface 到 telechat容器的/home下
- 例如 复制代码文件夹 Telechat 到 telechat容器的/home下

```shell
sudo docker cp telechat_opensource_7B_huggingface telechat:/home/. && sudo docker cp TeleChat telechat:/home/.
```

进入容器

```shell
sudo docker  exec -it telechat bash
```

![挂载模型和代码](../images/挂载模型代码.png)


## 模型推理

进入Telechat/inference_telechat

```shell
python3 telechat_infer_demo.py
```

![直接推理结果](../images/直接推理结果.png)

### 长文外推

我们通过使用NTK-aware外推和attention scaling的方法，能够将在8K长度上微调的模型在推理时外推到96K的长度。下表展示了TeleChat-7B模型在不同长度wikipedia数据上困惑度，可以观察到同时使用NTK-aware外推和attention scaling方法时，TeleChat在96K的推理长度上依然具有较低的困惑度。

|                                    | 2048   | 4096   | 8192    | 16384   | 32768    | 65536    | 98304    |
| ---------------------------------- | ------ | ------ | ------- | ------- | -------- | -------- | -------- |
| baseline                           | 4.8122 | 4.6562 | 39.3099 | 98.3102 | 155.2708 | 487.3398 | 447.6295 |
| NTK-aware (8k)                     | 4.8122 | 4.6562 | 5.1904  | 4.7155  | 8.6351   | 77.7478  | 79.9256  |
| NTK-aware+attention  scaling (8k)  | 4.8122 | 4.6562 | 5.1904  | 4.0353  | 4.1408   | 9.4080   | 7.9711   |
| NTK-aware (16k)                    | 7.6916 | 7.9900 | 7.9580  | 5.1217  | 4.7932   | 10.5444  | 10.3614  |
| NTK-aware+attention  scaling (16k) | 7.6916 | 7.9900 | 7.9580  | 5.1217  | 4.7195   | 8.9751   | 7.6822   |

当然，您也可以在更长的长度上微调TeleChat，使之具备更强的外推能力。微调之后，只需**将模型的`config.json`配置文件中的`training_seqlen`字段修改为微调时的训练长度**即可进行推理。上表的第4、5行展示了将TeleChat-7B在16K长度上微调之后的困惑度，观察到在64K以上的推理长度上具有更低的困惑度。

## 模型微调

模型微调分为全参数微调和lora微调两种方式：

### LoRA微调

**进入`deepspeed-telechat/sft`路径**， 按照下述命令运行，启动基于DeepSpeed LoRA微调。

```shell
bash run_telechat_lora.sh
```

### 全参数微调

**进入`deepspeed-telechat/sft`路径**，按照下述命令运行，启动基于DeepSpeed的全参数微调。

单节点运行脚本

```shell
bash run_telechat_single_node.sh
```

### 微调后推理测试

**进入`inference_telechat/`路径**，修改telechat_infer_demo.py中PATH为上一步保存的模型路径文件，随后，按照下述命令运行，进行模型的推理

```shell
python telechat_infer_demo.py
```

## 模型量化

### GPTQ量化

进入Telechat/quant

```shell
python quant.py
```

![量化结果](../images/量化结果.png)

### 量化推理

调用推理

```shell
python telechat_quantized_infer_demo.py
```

![量化推理结果](../images/量化推理结果.png)

## 服务化

提供了两种简单的**单并发场景**服务化方式，支持流式返回

### API

进入service 文件夹

```shell
python telechat_service.py
```
![API](../images/api页面.png)

默认在0.0.0.0:8070会启动telechat服务,可以使用test_json.py,test_stream.py进行测试

其它机器访问服务，需要修改0.0.0.0为服务机器IP。
### WEB

在完成API部署后，运行

```shell
streamlit run webdemo.py
```
![API](../images/web页面.png)

默认在0.0.0.0:8501

其它机器访问服务，需要修改0.0.0.0为服务机器IP。