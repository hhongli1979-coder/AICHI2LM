# 智能AI助手模块 (Intelligent AI Assistant)

## 概述 (Overview)

这是一个高智商、带记忆功能和语言理解能力的智能AI助手模块。该模块基于TeleChat大语言模型，扩展了以下核心能力：

- **记忆管理系统 (Memory Management)**: 支持短期记忆、长期记忆和知识库存储
- **推理引擎 (Reasoning Engine)**: 支持思维链推理、反思推理和类比推理
- **智能代理 (Intelligent Agent)**: 整合记忆和推理能力的对话代理

## 模块结构 (Module Structure)

```
intelligent_assistant/
├── __init__.py           # 模块入口
├── memory_manager.py     # 记忆管理器
├── reasoning_engine.py   # 推理引擎
├── intelligent_agent.py  # 智能代理
├── demo.py              # 演示脚本
└── README.md            # 文档
```

## 快速开始 (Quick Start)

### 安装依赖 (Dependencies)

```bash
# 无需额外依赖，使用Python标准库
python >= 3.7
```

### 基本使用 (Basic Usage)

```python
from intelligent_assistant import IntelligentAgent

# 创建智能代理
agent = IntelligentAgent(name="AICHI", personality="helpful")

# 对话
response = agent.chat("你好")
print(response)
```

## 详细文档 (Detailed Documentation)

### 1. 记忆管理器 (Memory Manager)

记忆管理器负责管理对话历史、知识存储和上下文记忆。

```python
from intelligent_assistant import MemoryManager

# 创建记忆管理器
memory = MemoryManager(
    short_term_capacity=100,    # 短期记忆容量
    long_term_capacity=1000,    # 长期记忆容量
    persistence_path="memory.json"  # 持久化路径
)

# 添加对话
memory.add_conversation("user", "你好，我是小明")
memory.add_conversation("bot", "你好小明！")

# 添加记忆
memory.add_memory(
    content="重要信息",
    memory_type="long_term",
    importance=8.0
)

# 添加知识
memory.add_knowledge("user_name", "小明")

# 搜索记忆
results = memory.search_memory("小明")

# 获取记忆摘要
summary = memory.get_memory_summary()

# 持久化保存
memory.save_to_file()
```

#### 记忆类型

| 类型 | 说明 | 容量 |
|------|------|------|
| short_term | 短期记忆，存储最近对话 | 默认100条 |
| long_term | 长期记忆，存储重要信息 | 默认1000条 |
| knowledge | 知识库，存储结构化知识 | 无限制 |

### 2. 推理引擎 (Reasoning Engine)

推理引擎提供高智商推理能力。

```python
from intelligent_assistant import ReasoningEngine

engine = ReasoningEngine()

# 思维链推理 (Chain of Thought)
result = engine.chain_of_thought(
    question="为什么天空是蓝色的？",
    context="光学相关背景知识"
)
print(result.answer)
print(result.get_explanation())

# 反思推理 (Reflection)
result = engine.reflect(
    question="原问题",
    initial_answer="初始答案",
    feedback="需要改进的地方"
)

# 类比推理 (Analogy)
result = engine.analogy_reasoning(
    source_situation="已知情况",
    target_situation="新情况",
    source_solution="已知解决方案"
)
```

#### 推理类型

| 类型 | 说明 |
|------|------|
| chain_of_thought | 思维链推理，逐步分析问题 |
| reflection | 反思推理，改进初始答案 |
| analogy | 类比推理，从已知推未知 |
| deduction | 演绎推理 |
| induction | 归纳推理 |

### 3. 智能代理 (Intelligent Agent)

智能代理整合了记忆管理和推理引擎，提供完整的对话能力。

```python
from intelligent_assistant import IntelligentAgent

# 创建代理
agent = IntelligentAgent(
    name="AICHI",           # 代理名称
    personality="helpful",   # 个性特征
    memory_path="agent_memory.json"  # 记忆持久化路径
)

# 对话
response = agent.chat("你好")
response = agent.chat("为什么AI很重要？", use_reasoning=True)

# 学习知识
agent.learn_knowledge("user_birthday", "10月15日")

# 回忆知识
birthday = agent.recall_knowledge("user_birthday")

# 对话反思
reflection = agent.reflect_on_conversation()

# 获取状态
status = agent.get_status()

# 注册自定义技能
def custom_skill(user_input, context):
    return "自定义回复"
agent.register_skill("custom", custom_skill)

# 保存状态
agent.save_state()
```

#### 内置技能

| 技能 | 触发词 | 功能 |
|------|--------|------|
| greeting | 你好, hello | 问候 |
| farewell | 再见, bye | 告别 |
| help | 帮助, help | 显示帮助信息 |
| memory_summary | 记忆摘要 | 显示记忆统计 |
| reasoning | 为什么, 怎么 | 推理分析 |

## 运行演示 (Run Demo)

```bash
cd AICHI2LM
python intelligent_assistant/demo.py
```

## API 参考 (API Reference)

### MemoryManager

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `add_memory()` | content, memory_type, importance | MemoryEntry | 添加记忆 |
| `add_conversation()` | role, content | None | 添加对话 |
| `search_memory()` | query, limit | List[MemoryEntry] | 搜索记忆 |
| `add_knowledge()` | key, value | None | 添加知识 |
| `get_knowledge()` | key | Any | 获取知识 |
| `get_memory_summary()` | - | Dict | 获取摘要 |
| `save_to_file()` | filepath | None | 保存到文件 |
| `load_from_file()` | filepath | bool | 从文件加载 |

### ReasoningEngine

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `chain_of_thought()` | question, context | ReasoningResult | 思维链推理 |
| `reflect()` | question, initial_answer, feedback | ReasoningResult | 反思推理 |
| `analogy_reasoning()` | source, target, solution | ReasoningResult | 类比推理 |

### IntelligentAgent

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `chat()` | user_input, use_reasoning | str | 对话 |
| `learn_knowledge()` | key, value | None | 学习知识 |
| `recall_knowledge()` | key | Any | 回忆知识 |
| `register_skill()` | name, handler | None | 注册技能 |
| `get_status()` | - | Dict | 获取状态 |
| `save_state()` | - | None | 保存状态 |

## 许可证 (License)

本模块遵循与TeleChat相同的许可协议。详见 [TeleChat模型社区许可协议](../TeleChat模型社区许可协议.pdf)
