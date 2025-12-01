# -*- coding: utf-8 -*-
"""
达尔文哥德尔机架构 - Darwin Godel Machine Architecture

实现基于进化算法的智能体种群管理和自我进化机制。
This implements an evolution-based agent population management
and self-improvement mechanism inspired by Darwin's natural selection
and Gödel machines' self-referential improvement.
"""

import random
import copy
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class Agent:
    """智能体基类 - Base Agent Class"""
    agent_id: str
    generation: int
    fitness_score: float = 0.0
    capabilities: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.capabilities:
            self.capabilities = {}
        if not self.metadata:
            self.metadata = {}


@dataclass
class EvolutionResult:
    """进化结果 - Evolution Result"""
    agent: Agent
    performance: float
    improvement: float
    generation: int


class AgentEvaluator(ABC):
    """智能体评估器接口 - Agent Evaluator Interface"""

    @abstractmethod
    def evaluate(self, agent: Agent, task: Any) -> float:
        """评估智能体在任务上的表现"""


class DefaultAgentEvaluator(AgentEvaluator):
    """默认智能体评估器"""

    def evaluate(self, agent: Agent, task: Any) -> float:
        """
        Default evaluation based on agent fitness score.
        In production, this should be replaced with actual benchmark evaluation.
        """
        return agent.fitness_score


class DarwinGodelMachine:
    """
    达尔文哥德尔机 - Darwin Godel Machine

    结合达尔文进化论和哥德尔机的自我改进机制,
    实现智能体种群的自动进化和优化。

    Combines Darwinian evolution with Gödel machine's self-improvement
    mechanism to enable automatic evolution and optimization of agent populations.
    """

    def __init__(
        self,
        population_size: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_ratio: float = 0.2,
        evaluator: Optional[AgentEvaluator] = None
    ):
        """
        初始化达尔文哥德尔机

        Args:
            population_size: 种群大小
            mutation_rate: 变异率
            crossover_rate: 交叉率
            elite_ratio: 精英保留比例
            evaluator: 智能体评估器
        """
        self.agent_pool: List[Agent] = []
        self.evolution_cycle: int = 0
        self.performance_history: List[Dict[str, float]] = []

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.evaluator = evaluator or DefaultAgentEvaluator()

        self._best_agent: Optional[Agent] = None
        self._generation_stats: List[Dict[str, Any]] = []

    def initialize_population(
        self,
        base_capabilities: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        初始化智能体种群

        Args:
            base_capabilities: 基础能力配置
        """
        base_capabilities = base_capabilities or {}
        self.agent_pool = []

        for i in range(self.population_size):
            agent = Agent(
                agent_id=f"agent_{self.evolution_cycle}_{i}",
                generation=self.evolution_cycle,
                fitness_score=random.random(),
                capabilities=copy.deepcopy(base_capabilities),
                metadata={"created_at": self.evolution_cycle}
            )
            self.agent_pool.append(agent)

    def select_best_agent(self) -> Agent:
        """
        从种群中选择最优智能体 (精英选择)

        Returns:
            最优智能体
        """
        if not self.agent_pool:
            raise ValueError("Agent pool is empty. Initialize population first.")

        # 按适应度排序
        sorted_agents = sorted(
            self.agent_pool,
            key=lambda a: a.fitness_score,
            reverse=True
        )
        return sorted_agents[0]

    def tournament_selection(self, tournament_size: int = 3) -> Agent:
        """
        锦标赛选择

        Args:
            tournament_size: 锦标赛规模

        Returns:
            选中的智能体
        """
        if not self.agent_pool:
            raise ValueError("Agent pool is empty.")

        tournament = random.sample(
            self.agent_pool,
            min(tournament_size, len(self.agent_pool))
        )
        return max(tournament, key=lambda a: a.fitness_score)

    def mutate_agent(self, parent: Agent) -> Agent:
        """
        变异智能体 - 通过随机扰动产生新版本

        Args:
            parent: 父代智能体

        Returns:
            变异后的新智能体
        """
        mutated = Agent(
            agent_id=f"agent_{self.evolution_cycle}_{len(self.agent_pool)}",
            generation=self.evolution_cycle,
            fitness_score=0.0,
            capabilities=copy.deepcopy(parent.capabilities),
            metadata={
                "parent_id": parent.agent_id,
                "mutation_type": "standard",
                "created_at": self.evolution_cycle
            }
        )

        # 对能力参数进行变异
        for key in mutated.capabilities:
            if random.random() < self.mutation_rate:
                if isinstance(mutated.capabilities[key], (int, float)):
                    # 数值类型: 添加高斯噪声
                    noise = random.gauss(0, 0.1)
                    mutated.capabilities[key] *= (1 + noise)
                elif isinstance(mutated.capabilities[key], bool):
                    # 布尔类型: 随机翻转
                    mutated.capabilities[key] = not mutated.capabilities[key]
                elif isinstance(mutated.capabilities[key], list):
                    # 列表类型: 随机打乱或添加/删除元素
                    if mutated.capabilities[key]:
                        random.shuffle(mutated.capabilities[key])

        return mutated

    def crossover(self, parent1: Agent, parent2: Agent) -> Agent:
        """
        交叉操作 - 结合两个父代的特性

        Args:
            parent1: 第一个父代
            parent2: 第二个父代

        Returns:
            子代智能体
        """
        child = Agent(
            agent_id=f"agent_{self.evolution_cycle}_{len(self.agent_pool)}",
            generation=self.evolution_cycle,
            fitness_score=0.0,
            capabilities={},
            metadata={
                "parent1_id": parent1.agent_id,
                "parent2_id": parent2.agent_id,
                "crossover_type": "uniform",
                "created_at": self.evolution_cycle
            }
        )

        # 均匀交叉
        all_keys = set(parent1.capabilities.keys()) | set(parent2.capabilities.keys())
        for key in all_keys:
            if key in parent1.capabilities and key in parent2.capabilities:
                # 两个父代都有该能力，随机选择
                child.capabilities[key] = copy.deepcopy(
                    parent1.capabilities[key] if random.random() < 0.5
                    else parent2.capabilities[key]
                )
            elif key in parent1.capabilities:
                child.capabilities[key] = copy.deepcopy(parent1.capabilities[key])
            else:
                child.capabilities[key] = copy.deepcopy(parent2.capabilities[key])

        return child

    def validate_agent(
        self,
        agent: Agent,
        benchmark_tasks: Optional[List[Any]] = None
    ) -> float:
        """
        在基准测试上验证智能体性能

        Args:
            agent: 待验证的智能体
            benchmark_tasks: 基准测试任务列表

        Returns:
            性能评分
        """
        if benchmark_tasks is None:
            # 使用默认评估
            performance = self.evaluator.evaluate(agent, None)
        else:
            # 在多个任务上评估并取平均
            scores = [
                self.evaluator.evaluate(agent, task)
                for task in benchmark_tasks
            ]
            performance = sum(scores) / len(scores) if scores else 0.0

        agent.fitness_score = performance
        return performance

    def update_pool(self, new_agent: Agent, performance: float) -> None:
        """
        优胜劣汰 - 更新种群

        Args:
            new_agent: 新智能体
            performance: 新智能体的性能评分
        """
        new_agent.fitness_score = performance

        if len(self.agent_pool) >= self.population_size:
            # 移除最差的智能体
            self.agent_pool.sort(key=lambda a: a.fitness_score)
            if new_agent.fitness_score > self.agent_pool[0].fitness_score:
                self.agent_pool[0] = new_agent
        else:
            self.agent_pool.append(new_agent)

        # 更新最佳智能体
        if self._best_agent is None or performance > self._best_agent.fitness_score:
            self._best_agent = copy.deepcopy(new_agent)

    def evolve(
        self,
        benchmark_tasks: Optional[List[Any]] = None
    ) -> EvolutionResult:
        """
        执行一次进化循环

        Args:
            benchmark_tasks: 基准测试任务列表

        Returns:
            进化结果
        """
        if not self.agent_pool:
            raise ValueError("Agent pool is empty. Initialize population first.")

        # 1. 选择最优智能体作为父代
        parent = self.select_best_agent()
        old_best_score = parent.fitness_score

        # 2. 决定是变异还是交叉
        if random.random() < self.crossover_rate and len(self.agent_pool) >= 2:
            # 交叉操作
            parent2 = self.tournament_selection()
            new_agent = self.crossover(parent, parent2)
            # 可能再进行变异
            if random.random() < self.mutation_rate:
                new_agent = self.mutate_agent(new_agent)
        else:
            # 纯变异操作
            new_agent = self.mutate_agent(parent)

        # 3. 验证新智能体
        performance = self.validate_agent(new_agent, benchmark_tasks)

        # 4. 更新种群
        self.update_pool(new_agent, performance)

        # 5. 记录历史
        improvement = performance - old_best_score
        self.performance_history.append({
            "cycle": self.evolution_cycle,
            "best_performance": self.select_best_agent().fitness_score,
            "avg_performance": sum(a.fitness_score for a in self.agent_pool) / len(self.agent_pool),
            "improvement": improvement
        })

        self.evolution_cycle += 1

        return EvolutionResult(
            agent=new_agent,
            performance=performance,
            improvement=improvement,
            generation=self.evolution_cycle
        )

    def evolve_generation(
        self,
        generations: int = 1,
        benchmark_tasks: Optional[List[Any]] = None
    ) -> List[EvolutionResult]:
        """
        进化多代

        Args:
            generations: 进化代数
            benchmark_tasks: 基准测试任务列表

        Returns:
            每代的进化结果列表
        """
        results = []
        for _ in range(generations):
            result = self.evolve(benchmark_tasks)
            results.append(result)
        return results

    def get_elite(self, n: int = 1) -> List[Agent]:
        """
        获取精英智能体

        Args:
            n: 精英数量

        Returns:
            精英智能体列表
        """
        sorted_pool = sorted(
            self.agent_pool,
            key=lambda a: a.fitness_score,
            reverse=True
        )
        return sorted_pool[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取进化统计信息

        Returns:
            统计信息字典
        """
        if not self.agent_pool:
            return {"error": "No agents in pool"}

        fitness_scores = [a.fitness_score for a in self.agent_pool]
        return {
            "evolution_cycle": self.evolution_cycle,
            "population_size": len(self.agent_pool),
            "best_fitness": max(fitness_scores),
            "worst_fitness": min(fitness_scores),
            "avg_fitness": sum(fitness_scores) / len(fitness_scores),
            "best_agent_id": self.select_best_agent().agent_id,
            "history_length": len(self.performance_history)
        }


class SelfEvolvingAgent(Agent):
    """
    自进化智能体 - Self-Evolving Agent

    具有自我改进能力的智能体，可以分析自身表现并进行自我优化
    """

    def __init__(
        self,
        agent_id: str,
        generation: int,
        fitness_score: float = 0.0,
        capabilities: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            agent_id=agent_id,
            generation=generation,
            fitness_score=fitness_score,
            capabilities=capabilities or {},
            metadata=metadata or {}
        )
        self.learning_history: List[Dict[str, Any]] = []
        self.self_improvement_count: int = 0

    def analyze_performance(self, task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析任务表现，识别改进点

        Args:
            task_results: 任务结果列表

        Returns:
            分析报告
        """
        analysis = {
            "total_tasks": len(task_results),
            "success_rate": 0.0,
            "weak_areas": [],
            "strong_areas": [],
            "improvement_suggestions": []
        }

        if not task_results:
            return analysis

        successes = sum(1 for r in task_results if r.get("success", False))
        analysis["success_rate"] = successes / len(task_results)

        # 分析弱项和强项
        for result in task_results:
            if result.get("score", 0) < 0.5:
                analysis["weak_areas"].append(result.get("task_type", "unknown"))
            elif result.get("score", 0) > 0.8:
                analysis["strong_areas"].append(result.get("task_type", "unknown"))

        return analysis

    def self_improve(self, analysis: Dict[str, Any]) -> None:
        """
        基于分析结果进行自我改进

        Args:
            analysis: 性能分析报告
        """
        # 记录改进历史
        self.learning_history.append({
            "cycle": self.self_improvement_count,
            "analysis": analysis,
            "action": "self_improvement"
        })

        # 针对弱项调整能力参数
        for weak_area in analysis.get("weak_areas", []):
            if weak_area in self.capabilities:
                # 增强该领域的能力
                if isinstance(self.capabilities[weak_area], (int, float)):
                    self.capabilities[weak_area] *= 1.1

        self.self_improvement_count += 1
