# -*- coding: utf-8 -*-
"""
神经进化架构使用示例 - Neural Evolution Architecture Examples

这个文件演示了如何使用神经进化系统的各个组件。
This file demonstrates how to use the neural evolution system components.
"""

import sys
import os

# Add the parent directory to path to import neural_evolution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_evolution import (
    DarwinGodelMachine,
    EvolutionaryMemory,
    ToolEvolutionSystem,
    MultiRoundThinking,
    SelfRewardingSystem,
    EvolutionMonitor,
    EvolutionaryLaws,
    UnifiedMultimodalBrain,
    VoiceNeuralNetwork
)
from neural_evolution.integration import (
    NeuralEvolutionSystem,
    EvolutionConfig,
    create_neural_evolution_system
)
from neural_evolution.evolutionary_memory import Experience
from neural_evolution.evolution_laws import Modification, ModificationType
from neural_evolution.evolution_monitor import MetricType


def demo_darwin_godel_machine():
    """演示达尔文哥德尔机"""
    print("=" * 50)
    print("达尔文哥德尔机演示 - Darwin Gödel Machine Demo")
    print("=" * 50)

    # 创建达尔文哥德尔机
    machine = DarwinGodelMachine(
        population_size=5,
        mutation_rate=0.1,
        crossover_rate=0.7
    )

    # 初始化种群
    machine.initialize_population({
        "reasoning_ability": 0.5,
        "learning_speed": 0.5,
        "creativity": 0.5
    })

    print(f"初始种群大小: {len(machine.agent_pool)}")
    print(f"初始最佳适应度: {machine.select_best_agent().fitness_score:.4f}")

    # 执行进化
    for i in range(5):
        result = machine.evolve()
        print(f"第 {i + 1} 代: 性能={result.performance:.4f}, 改进={result.improvement:.4f}")

    print(f"\n进化统计: {machine.get_statistics()}")


def demo_evolutionary_memory():
    """演示进化记忆系统"""
    print("\n" + "=" * 50)
    print("进化记忆系统演示 - Evolutionary Memory Demo")
    print("=" * 50)

    memory = EvolutionaryMemory()

    # 添加一些经验
    experiences = [
        Experience(
            task_type="coding",
            input_data="写一个排序函数",
            output_data="def sort(arr): return sorted(arr)",
            success=True,
            score=0.8,
            metadata={"strategy": "divide_and_conquer"}
        ),
        Experience(
            task_type="coding",
            input_data="实现二分查找",
            output_data="def binary_search(arr, x): ...",
            success=True,
            score=0.9,
            metadata={"approach": "recursive"}
        ),
        Experience(
            task_type="writing",
            input_data="写一篇技术博客",
            output_data="[博客内容...]",
            success=True,
            score=0.75,
            metadata={"strategy": "structured_outline"}
        )
    ]

    for exp in experiences:
        insight = memory.evolve_memory(exp)
        if insight:
            print(f"从经验中提炼出洞察: {insight.content}")

    # 查询洞察
    coding_insights = memory.query_insights(topic="coding", limit=5)
    print(f"\n编程相关洞察数量: {len(coding_insights)}")

    # 获取相关上下文
    context = memory.get_relevant_context("coding")
    print(f"相关上下文: {context}")

    print(f"\n记忆系统统计: {memory.get_statistics()}")


def demo_multi_round_thinking():
    """演示多轮思考训练"""
    print("\n" + "=" * 50)
    print("多轮思考训练演示 - Multi-Round Thinking Demo")
    print("=" * 50)

    thinking = MultiRoundThinking(
        thinking_rounds=3,
        quality_threshold=0.7
    )

    problem = "如何设计一个高效的缓存系统？"

    best_solution, all_solutions = thinking.train_self(problem)

    print(f"问题: {problem}")
    print(f"思考轮数: {len(all_solutions)}")
    print(f"最终质量: {best_solution.quality.value}")
    print(f"最终得分: {best_solution.score:.4f}")
    print(f"\n推理步骤:")
    for step in best_solution.reasoning_steps:
        print(f"  - {step}")

    print(f"\n思考系统统计: {thinking.get_statistics()}")


def demo_self_reward():
    """演示自我奖励系统"""
    print("\n" + "=" * 50)
    print("自我奖励系统演示 - Self-Rewarding System Demo")
    print("=" * 50)

    reward_system = SelfRewardingSystem(
        learning_rate=0.1,
        reward_threshold=0.7
    )

    # 评估任务表现
    tasks = [
        ("写一个Python函数计算阶乘", "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"),
        ("解释什么是递归", "递归是一种函数调用自身的编程技术..."),
        ("翻译这段话", "[翻译结果]")
    ]

    for task, solution in tasks:
        score, reward = reward_system.evaluate_own_performance(task, solution)
        print(f"\n任务: {task[:30]}...")
        print(f"  评分: {score:.4f}")
        print(f"  奖励类型: {reward.reward_type.value}")
        print(f"  奖励值: {reward.value:.4f}")
        print(f"  原因: {reward.reason}")

        # 进行强化学习
        update = reward_system.reinforce_learning(reward)
        print(f"  学习更新: TD误差={update['td_error']:.4f}")

    print(f"\n自我奖励系统统计: {reward_system.get_statistics()}")


def demo_evolution_monitor():
    """演示进化监控系统"""
    print("\n" + "=" * 50)
    print("进化监控系统演示 - Evolution Monitor Demo")
    print("=" * 50)

    monitor = EvolutionMonitor()

    # 记录一些指标
    for i in range(10):
        monitor.record_metric(MetricType.INTELLIGENCE, 0.5 + i * 0.03)
        monitor.record_metric(MetricType.LEARNING_SPEED, 0.4 + i * 0.05)
        monitor.record_metric(MetricType.CREATIVITY, 0.6 + i * 0.02)

    # 跟踪进化
    metrics = monitor.track_evolution()
    print(f"当前进化指标:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # 获取趋势
    trend = monitor.get_trend(MetricType.INTELLIGENCE)
    print(f"\n智力指标趋势: {trend}")

    print(f"\n当前策略: {monitor.current_strategy.name}")
    print(f"策略参数: {monitor.get_strategy_parameters()}")


def demo_evolution_laws():
    """演示进化安全定律"""
    print("\n" + "=" * 50)
    print("进化安全定律演示 - Evolution Laws Demo")
    print("=" * 50)

    laws = EvolutionaryLaws(
        safety_threshold=0.8,
        performance_threshold=0.0
    )

    # 测试安全修改
    safe_modification = Modification(
        modification_id="mod_001",
        modification_type=ModificationType.PARAMETER_CHANGE,
        description="提升学习率",
        changes={"learning_rate": 0.01},
        expected_impact=0.05,
        metadata={"rollback_procedure": "恢复原学习率"}
    )

    print(f"测试安全修改: {safe_modification.description}")
    allowed, result = laws.law3_evolve(safe_modification)
    print(f"  允许进化: {allowed}")
    print(f"  阶段: {result['stage']}")

    # 测试危险修改
    risky_modification = Modification(
        modification_id="mod_002",
        modification_type=ModificationType.ARCHITECTURE_CHANGE,
        description="攻击性系统重构",
        changes={"memory_increase": 0.8},
        expected_impact=-0.1
    )

    print(f"\n测试危险修改: {risky_modification.description}")
    allowed, result = laws.law3_evolve(risky_modification)
    print(f"  允许进化: {allowed}")
    print(f"  阶段: {result['stage']}")

    print(f"\n安全定律统计: {laws.get_statistics()}")


def demo_multimodal_brain():
    """演示多模态大脑"""
    print("\n" + "=" * 50)
    print("多模态大脑演示 - Multimodal Brain Demo")
    print("=" * 50)

    brain = UnifiedMultimodalBrain()

    # 处理文本输入
    text_result = brain.process_text("这是一段测试文本")
    print(f"文本处理结果:")
    print(f"  模态: {[m.value for m in text_result.modalities]}")
    print(f"  对齐分数: {text_result.alignment_score:.4f}")

    # 处理多模态输入
    multimodal_result = brain.process_multimodal(
        text="这是图片描述",
        image="[图像数据]"
    )
    print(f"\n多模态处理结果:")
    print(f"  模态: {[m.value for m in multimodal_result.modalities]}")
    print(f"  对齐分数: {multimodal_result.alignment_score:.4f}")

    # 语音网络演示
    voice_network = brain.voice_network
    synthesis = voice_network.synthesize_speech("你好，我是TeleChat")
    print(f"\n语音合成结果:")
    print(f"  文本: {synthesis['text']}")
    print(f"  时长: {synthesis['duration']:.2f}秒")
    print(f"  情感: {synthesis['emotion']}")

    print(f"\n多模态大脑统计: {brain.get_statistics()}")


def demo_integrated_system():
    """演示集成系统"""
    print("\n" + "=" * 50)
    print("神经进化集成系统演示 - Integrated System Demo")
    print("=" * 50)

    # 创建集成系统
    system = create_neural_evolution_system(
        population_size=5,
        thinking_rounds=2,
        safety_threshold=0.7
    )

    print("系统初始状态:")
    status = system.get_system_status()
    print(f"  已初始化: {status['initialized']}")
    print(f"  进化次数: {status['evolution_count']}")

    # 执行进化
    print("\n执行进化循环...")
    result = system.evolve("如何优化数据库查询性能？")

    print(f"进化结果:")
    print(f"  达尔文机性能: {result['components']['darwin']['performance']:.4f}")
    if 'thinking' in result['components']:
        print(f"  思考轮数: {result['components']['thinking']['rounds']}")
        print(f"  思考质量: {result['components']['thinking']['quality']}")

    # 思考并解决问题
    print("\n思考解决问题...")
    solution_result = system.think_and_solve(
        "设计一个分布式缓存系统",
        max_rounds=3
    )
    print(f"解决方案得分: {solution_result['score']:.4f}")
    print(f"解决方案质量: {solution_result['quality']}")

    # 从经验中学习
    print("\n从经验中学习...")
    learn_result = system.learn_from_experience(
        task="代码审查任务",
        result="发现3个潜在bug",
        success=True,
        score=0.85
    )
    print(f"  经验已添加: {learn_result['experience_added']}")
    print(f"  创建洞察: {learn_result['insight_created']}")

    # 获取进化指标
    print("\n当前进化指标:")
    metrics = system.get_evolution_metrics()
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\n最终系统状态:")
    final_status = system.get_system_status()
    print(f"  进化次数: {final_status['evolution_count']}")


def main():
    """主函数 - 运行所有演示"""
    print("神经进化架构完整演示")
    print("Neural Evolution Architecture Complete Demo")
    print("=" * 60)

    demo_darwin_godel_machine()
    demo_evolutionary_memory()
    demo_multi_round_thinking()
    demo_self_reward()
    demo_evolution_monitor()
    demo_evolution_laws()
    demo_multimodal_brain()
    demo_integrated_system()

    print("\n" + "=" * 60)
    print("演示完成！所有神经进化组件均可正常工作。")
    print("Demo completed! All neural evolution components work correctly.")


if __name__ == "__main__":
    main()
