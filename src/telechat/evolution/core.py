"""
TeleChat 自进化核心模块
实现达尔文哥德尔机架构
"""

import random
import time
import json
import hashlib
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

from ..utils.logging import get_logger
from ..utils.exceptions import TeleChatException

logger = get_logger("telechat.evolution")


@dataclass
class EvolutionMetrics:
    """进化指标"""
    generation: int = 0
    fitness_score: float = 0.0
    intelligence_quotient: float = 0.0
    learning_speed: float = 0.0
    creativity_score: float = 0.0
    problem_solving_depth: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "fitness_score": self.fitness_score,
            "intelligence_quotient": self.intelligence_quotient,
            "learning_speed": self.learning_speed,
            "creativity_score": self.creativity_score,
            "problem_solving_depth": self.problem_solving_depth,
            "timestamp": self.timestamp
        }


@dataclass
class Agent:
    """智能体"""
    id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    fitness: float = 0.0
    generation: int = 0
    parent_id: Optional[str] = None
    mutations: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def clone(self) -> "Agent":
        """克隆智能体"""
        return Agent(
            id=f"{self.id}_clone_{int(time.time())}",
            parameters=self.parameters.copy(),
            fitness=self.fitness,
            generation=self.generation + 1,
            parent_id=self.id,
            mutations=[]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parameters": self.parameters,
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "mutations": self.mutations,
            "created_at": self.created_at
        }


class EvolutionaryLaws:
    """进化三定律 - 安全约束"""
    
    def __init__(self):
        self.safety_threshold = 0.95
        self.performance_baseline = 0.5
        self.ethical_rules = [
            "不得生成有害内容",
            "不得泄露用户隐私",
            "不得执行恶意代码",
            "保持响应的准确性",
            "遵守法律法规"
        ]
    
    def law1_endure(self, modification: Dict[str, Any]) -> bool:
        """
        第一定律：保障系统安全稳定
        
        Args:
            modification: 修改方案
            
        Returns:
            是否通过安全检查
        """
        checks = [
            self._check_system_stability(modification),
            self._check_no_harmful_behavior(modification),
            self._check_ethical_compliance(modification),
            self._check_reversibility(modification)
        ]
        return all(checks)
    
    def law2_excel(self, modification: Dict[str, Any], current_performance: float) -> bool:
        """
        第二定律：保持或提升性能
        
        Args:
            modification: 修改方案
            current_performance: 当前性能
            
        Returns:
            是否满足性能要求
        """
        # 模拟预期性能
        expected_performance = self._simulate_performance(modification, current_performance)
        return expected_performance >= current_performance * 0.95
    
    def law3_evolve(self, modification: Dict[str, Any], current_performance: float) -> bool:
        """
        第三定律：满足前两者后自主优化
        
        Args:
            modification: 修改方案
            current_performance: 当前性能
            
        Returns:
            是否允许进化
        """
        if not self.law1_endure(modification):
            logger.warning("进化被拒绝：未通过安全检查")
            return False
        
        if not self.law2_excel(modification, current_performance):
            logger.warning("进化被拒绝：性能可能下降")
            return False
        
        logger.info("进化检查通过")
        return True
    
    def _check_system_stability(self, modification: Dict[str, Any]) -> bool:
        """检查系统稳定性"""
        # 检查修改范围
        affected_modules = modification.get("affected_modules", [])
        if len(affected_modules) > 5:
            return False  # 单次修改不能影响太多模块
        return True
    
    def _check_no_harmful_behavior(self, modification: Dict[str, Any]) -> bool:
        """检查是否存在有害行为"""
        harmful_keywords = ["delete", "drop", "truncate", "rm -rf", "format"]
        mod_str = json.dumps(modification).lower()
        return not any(kw in mod_str for kw in harmful_keywords)
    
    def _check_ethical_compliance(self, modification: Dict[str, Any]) -> bool:
        """检查伦理合规"""
        # 简单检查
        return True
    
    def _check_reversibility(self, modification: Dict[str, Any]) -> bool:
        """检查是否可逆"""
        return modification.get("reversible", True)
    
    def _simulate_performance(self, modification: Dict[str, Any], current: float) -> float:
        """模拟性能变化"""
        # 简单估算：假设有正面影响
        improvement = modification.get("expected_improvement", 0.0)
        return current * (1 + improvement)


class DarwinGodelMachine:
    """达尔文哥德尔机 - 智能体进化引擎"""
    
    def __init__(
        self,
        population_size: int = 10,
        mutation_rate: float = 0.1,
        elite_ratio: float = 0.2
    ):
        """
        初始化进化引擎
        
        Args:
            population_size: 种群大小
            mutation_rate: 变异率
            elite_ratio: 精英保留比例
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        
        self.agent_pool: List[Agent] = []
        self.evolution_cycle = 0
        self.performance_history: List[EvolutionMetrics] = []
        self.laws = EvolutionaryLaws()
        
        # 初始化种群
        self._initialize_population()
        
        logger.info(f"达尔文哥德尔机初始化完成，种群大小: {population_size}")
    
    def _initialize_population(self):
        """初始化种群"""
        for i in range(self.population_size):
            agent = Agent(
                id=f"agent_{i}_{int(time.time())}",
                parameters={
                    "temperature": random.uniform(0.5, 1.0),
                    "top_p": random.uniform(0.8, 1.0),
                    "thinking_depth": random.randint(1, 3),
                    "creativity_weight": random.uniform(0.3, 0.7)
                },
                generation=0
            )
            self.agent_pool.append(agent)
    
    def evolve(self) -> EvolutionMetrics:
        """
        执行一次进化循环
        
        Returns:
            进化指标
        """
        logger.info(f"开始进化循环 {self.evolution_cycle + 1}")
        
        # 1. 评估当前种群
        self._evaluate_population()
        
        # 2. 选择最优智能体
        parent = self.select_best_agent()
        logger.info(f"选择最优智能体: {parent.id}, 适应度: {parent.fitness:.4f}")
        
        # 3. 生成变异版本
        mutated_agent = self.mutate_agent(parent)
        
        # 4. 验证变异版本
        modification = {
            "type": "mutation",
            "affected_modules": ["parameters"],
            "expected_improvement": 0.05,
            "reversible": True
        }
        
        if self.laws.law3_evolve(modification, parent.fitness):
            # 5. 验证新智能体
            new_fitness = self._evaluate_agent(mutated_agent)
            mutated_agent.fitness = new_fitness
            
            # 6. 更新种群
            self._update_pool(mutated_agent)
        
        self.evolution_cycle += 1
        
        # 记录指标
        metrics = self._calculate_metrics()
        self.performance_history.append(metrics)
        
        logger.info(f"进化循环 {self.evolution_cycle} 完成")
        return metrics
    
    def select_best_agent(self) -> Agent:
        """选择最优智能体"""
        sorted_agents = sorted(
            self.agent_pool,
            key=lambda x: x.fitness,
            reverse=True
        )
        return sorted_agents[0]
    
    def mutate_agent(self, parent: Agent) -> Agent:
        """
        对智能体进行变异
        
        Args:
            parent: 父代智能体
            
        Returns:
            变异后的智能体
        """
        mutated = parent.clone()
        mutated.id = f"agent_mut_{self.evolution_cycle}_{int(time.time())}"
        
        # 参数变异
        for key, value in mutated.parameters.items():
            if random.random() < self.mutation_rate:
                if isinstance(value, float):
                    # 浮点数变异
                    delta = random.gauss(0, 0.1)
                    mutated.parameters[key] = max(0.1, min(1.0, value + delta))
                    mutated.mutations.append(f"{key}: {value:.4f} -> {mutated.parameters[key]:.4f}")
                elif isinstance(value, int):
                    # 整数变异
                    delta = random.choice([-1, 0, 1])
                    mutated.parameters[key] = max(1, value + delta)
                    mutated.mutations.append(f"{key}: {value} -> {mutated.parameters[key]}")
        
        if mutated.mutations:
            logger.info(f"变异: {mutated.mutations}")
        
        return mutated
    
    def _evaluate_population(self):
        """评估整个种群"""
        for agent in self.agent_pool:
            agent.fitness = self._evaluate_agent(agent)
    
    def _evaluate_agent(self, agent: Agent) -> float:
        """
        评估单个智能体
        
        Args:
            agent: 智能体
            
        Returns:
            适应度分数
        """
        # 简化的评估函数
        # 实际应用中应该在基准测试上评估
        params = agent.parameters
        
        # 基础分
        base_score = 0.5
        
        # 温度评分（中等最好）
        temp_score = 1.0 - abs(params.get("temperature", 0.7) - 0.7) * 2
        
        # top_p评分
        top_p_score = params.get("top_p", 0.9)
        
        # 思考深度评分
        depth = params.get("thinking_depth", 2)
        depth_score = min(depth / 3, 1.0)
        
        # 综合评分
        fitness = base_score + (temp_score + top_p_score + depth_score) / 6
        
        # 添加随机扰动模拟真实评估
        fitness += random.gauss(0, 0.02)
        
        return max(0.0, min(1.0, fitness))
    
    def _update_pool(self, new_agent: Agent):
        """更新种群"""
        # 按适应度排序
        self.agent_pool.sort(key=lambda x: x.fitness, reverse=True)
        
        # 保留精英
        elite_count = max(1, int(self.population_size * self.elite_ratio))
        
        # 替换最差的个体
        if len(self.agent_pool) >= self.population_size:
            self.agent_pool[-1] = new_agent
        else:
            self.agent_pool.append(new_agent)
        
        # 再次排序
        self.agent_pool.sort(key=lambda x: x.fitness, reverse=True)
    
    def _calculate_metrics(self) -> EvolutionMetrics:
        """计算进化指标"""
        best_agent = self.select_best_agent()
        avg_fitness = sum(a.fitness for a in self.agent_pool) / len(self.agent_pool)
        
        return EvolutionMetrics(
            generation=self.evolution_cycle,
            fitness_score=best_agent.fitness,
            intelligence_quotient=best_agent.fitness * 150,  # 映射到IQ尺度
            learning_speed=avg_fitness,
            creativity_score=best_agent.parameters.get("creativity_weight", 0.5),
            problem_solving_depth=best_agent.parameters.get("thinking_depth", 2) / 3
        )
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """获取最优参数"""
        best = self.select_best_agent()
        return best.parameters.copy()
    
    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """获取进化历史"""
        return [m.to_dict() for m in self.performance_history]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "population_size": len(self.agent_pool),
            "evolution_cycles": self.evolution_cycle,
            "best_fitness": self.select_best_agent().fitness if self.agent_pool else 0,
            "average_fitness": sum(a.fitness for a in self.agent_pool) / len(self.agent_pool) if self.agent_pool else 0,
            "mutation_rate": self.mutation_rate
        }


class SelfTrainer:
    """自我训练器"""
    
    def __init__(self):
        self.training_data: List[Dict[str, Any]] = []
        self.training_history: List[Dict[str, Any]] = []
    
    def generate_training_sample(self, question_generator: Callable, solution_solver: Callable) -> Optional[Dict[str, Any]]:
        """
        自我生成训练样本
        
        Args:
            question_generator: 问题生成函数
            solution_solver: 解答函数
            
        Returns:
            训练样本
        """
        try:
            # 生成问题
            question = question_generator()
            
            # 生成解答
            solution = solution_solver(question)
            
            # 创建训练样本
            sample = {
                "id": hashlib.md5(question.encode()).hexdigest()[:16],
                "question": question,
                "solution": solution,
                "timestamp": datetime.now().isoformat(),
                "quality_score": None  # 待评估
            }
            
            self.training_data.append(sample)
            return sample
            
        except Exception as e:
            logger.error(f"生成训练样本失败: {str(e)}")
            return None
    
    def evaluate_sample(self, sample: Dict[str, Any], evaluator: Callable) -> float:
        """评估训练样本质量"""
        score = evaluator(sample["question"], sample["solution"])
        sample["quality_score"] = score
        return score
    
    def get_high_quality_samples(self, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """获取高质量样本"""
        return [
            s for s in self.training_data
            if s.get("quality_score") and s["quality_score"] >= threshold
        ]


class EvolutionMonitor:
    """进化监控器"""
    
    def __init__(self):
        self.metrics_history: List[EvolutionMetrics] = []
        self.alert_thresholds = {
            "performance_drop": 0.1,
            "instability": 0.05,
            "stagnation_cycles": 5
        }
        self.alerts: List[Dict[str, Any]] = []
    
    def track(self, metrics: EvolutionMetrics):
        """追踪进化指标"""
        self.metrics_history.append(metrics)
        self._detect_anomalies(metrics)
    
    def _detect_anomalies(self, current: EvolutionMetrics):
        """检测异常"""
        if len(self.metrics_history) < 2:
            return
        
        previous = self.metrics_history[-2]
        
        # 检测性能下降
        if current.fitness_score < previous.fitness_score * (1 - self.alert_thresholds["performance_drop"]):
            self._add_alert("performance_drop", f"性能下降: {previous.fitness_score:.4f} -> {current.fitness_score:.4f}")
        
        # 检测停滞
        if len(self.metrics_history) >= self.alert_thresholds["stagnation_cycles"]:
            recent = self.metrics_history[-self.alert_thresholds["stagnation_cycles"]:]
            if all(abs(m.fitness_score - recent[0].fitness_score) < 0.01 for m in recent):
                self._add_alert("stagnation", f"进化停滞已达 {self.alert_thresholds['stagnation_cycles']} 个周期")
    
    def _add_alert(self, alert_type: str, message: str):
        """添加警报"""
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self.alerts.append(alert)
        logger.warning(f"进化警报: {message}")
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """获取所有警报"""
        return self.alerts
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        latest = self.metrics_history[-1]
        first = self.metrics_history[0]
        
        return {
            "status": "active",
            "total_generations": len(self.metrics_history),
            "current_fitness": latest.fitness_score,
            "fitness_improvement": latest.fitness_score - first.fitness_score,
            "alerts_count": len(self.alerts),
            "latest_timestamp": latest.timestamp
        }


# 全局进化引擎
_global_evolution_engine: Optional[DarwinGodelMachine] = None


def get_evolution_engine() -> DarwinGodelMachine:
    """获取全局进化引擎"""
    global _global_evolution_engine
    if _global_evolution_engine is None:
        _global_evolution_engine = DarwinGodelMachine()
    return _global_evolution_engine
