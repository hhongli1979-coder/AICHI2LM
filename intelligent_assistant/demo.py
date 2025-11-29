# -*- coding: utf-8 -*-
"""
智能AI助手演示 - Intelligent AI Assistant Demo
展示高智商、记忆和语言能力
Demonstrates high-IQ, memory, and language capabilities
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelligent_assistant import IntelligentAgent, MemoryManager, ReasoningEngine


def demo_memory_manager():
    """演示记忆管理器功能 - Demo memory manager"""
    print("=" * 60)
    print("📦 记忆管理器演示 (Memory Manager Demo)")
    print("=" * 60)
    
    # 创建记忆管理器
    memory = MemoryManager()
    
    # 添加对话
    print("\n添加对话记录...")
    memory.add_conversation("user", "你好，我叫小明")
    memory.add_conversation("bot", "你好小明，很高兴认识你！")
    memory.add_conversation("user", "我喜欢编程和人工智能")
    memory.add_conversation("bot", "太棒了！我也对AI很感兴趣。")
    
    # 添加重要记忆
    print("添加重要记忆...")
    memory.add_memory(
        content="用户小明对AI和编程有浓厚兴趣",
        memory_type="long_term",
        importance=9.0,
        metadata={"category": "user_preference"}
    )
    
    # 添加知识
    print("添加知识到知识库...")
    memory.add_knowledge("user_name", "小明")
    memory.add_knowledge("user_interests", ["编程", "人工智能", "机器学习"])
    
    # 获取记忆摘要
    print("\n记忆摘要:")
    summary = memory.get_memory_summary()
    print(f"  - 短期记忆: {summary['short_term_count']} 条")
    print(f"  - 长期记忆: {summary['long_term_count']} 条")
    print(f"  - 知识库: {summary['knowledge_count']} 项")
    print(f"  - 对话记录: {summary['conversation_count']} 条")
    
    # 搜索记忆
    print("\n搜索记忆 '小明':")
    results = memory.search_memory("小明")
    for r in results:
        print(f"  - {r.content[:50]}...")
    
    # 获取最近上下文
    print("\n最近对话上下文:")
    context = memory.get_recent_context(3)
    for c in context:
        print(f"  [{c['role']}]: {c['content']}")
    
    return memory


def demo_reasoning_engine():
    """演示推理引擎功能 - Demo reasoning engine"""
    print("\n" + "=" * 60)
    print("🧠 推理引擎演示 (Reasoning Engine Demo)")
    print("=" * 60)
    
    engine = ReasoningEngine()
    
    # 思维链推理
    print("\n1. 思维链推理 (Chain of Thought):")
    question = "为什么天空是蓝色的？"
    result = engine.chain_of_thought(question)
    print(f"  问题: {question}")
    print(f"  答案: {result.answer}")
    print(f"  置信度: {result.total_confidence:.2f}")
    print(f"  推理步骤数: {len(result.reasoning_steps)}")
    
    # 反思推理
    print("\n2. 反思推理 (Reflection):")
    initial_answer = "天空是蓝色的因为阳光散射"
    reflection_result = engine.reflect(
        question=question,
        initial_answer=initial_answer,
        feedback="需要更详细的科学解释"
    )
    print(f"  初始答案: {initial_answer}")
    print(f"  改进后答案: {reflection_result.answer}")
    
    # 类比推理
    print("\n3. 类比推理 (Analogy):")
    analogy_result = engine.analogy_reasoning(
        source_situation="学习骑自行车需要练习和保持平衡",
        target_situation="学习编程",
        source_solution="通过反复练习逐步掌握技能"
    )
    print(f"  类比结果: {analogy_result.answer}")
    
    return engine


def demo_intelligent_agent():
    """演示智能代理功能 - Demo intelligent agent"""
    print("\n" + "=" * 60)
    print("🤖 智能代理演示 (Intelligent Agent Demo)")
    print("=" * 60)
    
    # 创建智能代理
    agent = IntelligentAgent(
        name="AICHI",
        personality="helpful"
    )
    
    # 模拟对话
    print("\n开始对话演示...\n")
    
    conversations = [
        "你好",
        "你能做什么？",
        "帮我记住，我的生日是10月15日",
        "为什么人工智能很重要？",
        "我之前说我的生日是什么时候？",
        "再见"
    ]
    
    for user_input in conversations:
        print(f"👤 用户: {user_input}")
        response = agent.chat(user_input)
        print(f"🤖 AICHI: {response}")
        print()
    
    # 显示代理状态
    print("\n代理状态:")
    status = agent.get_status()
    print(f"  - 名称: {status['name']}")
    print(f"  - 个性: {status['personality']}")
    print(f"  - 技能: {', '.join(status['skills'])}")
    print(f"  - 对话状态: {status['conversation_state']}")
    
    # 对话反思
    print("\n" + agent.reflect_on_conversation())
    
    return agent


def interactive_mode():
    """交互模式 - Interactive mode"""
    print("\n" + "=" * 60)
    print("💬 交互模式 (Interactive Mode)")
    print("=" * 60)
    print("输入 'quit' 或 '退出' 结束对话")
    print()
    
    agent = IntelligentAgent(name="AICHI", personality="friendly")
    
    while True:
        try:
            user_input = input("👤 你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "退出", "再见"]:
                response = agent.chat("再见")
                print(f"🤖 AICHI: {response}")
                break
            
            response = agent.chat(user_input)
            print(f"🤖 AICHI: {response}")
            print()
            
        except KeyboardInterrupt:
            print("\n\n感谢使用，再见！")
            break
        except EOFError:
            break


def main():
    """主函数 - Main function"""
    print("=" * 60)
    print("  🌟 AICHI - 智能AI助手 (Intelligent AI Assistant)")
    print("  具备高智商、记忆功能和语言理解能力")
    print("  High-IQ, Memory, and Language Capabilities")
    print("=" * 60)
    
    # 演示各个模块
    demo_memory_manager()
    demo_reasoning_engine()
    demo_intelligent_agent()
    
    # 询问是否进入交互模式
    print("\n" + "-" * 60)
    print("演示完成！")
    print("-" * 60)


if __name__ == "__main__":
    main()
