"""
进化算法实现 (Evolutionary Algorithm)
=====================================

实现基于种群的进化算法:
- 种群初始化
- 适应度评估
- 选择机制
- 交叉和变异
- 自然选择
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import random
import copy
import math

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """智能体数据类"""
    agent_id: str
    genome: Dict[str, Any]  # 基因组/参数
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'agent_id': self.agent_id,
            'genome': self.genome,
            'fitness': self.fitness,
            'generation': self.generation,
            'parent_ids': self.parent_ids
        }


@dataclass
class EvolutionConfig:
    """进化配置数据类"""
    population_size: int = 100
    elite_ratio: float = 0.1
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    tournament_size: int = 5
    max_generations: int = 1000
    fitness_threshold: float = 0.95
    stagnation_limit: int = 50


class EvolutionaryAlgorithm:
    """
    进化算法实现
    
    实现完整的进化循环:
    1. 种群初始化
    2. 适应度评估
    3. 选择优秀个体
    4. 交叉和变异
    5. 环境选择/自然选择
    """
    
    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        fitness_function: Optional[Callable[[Agent], float]] = None
    ):
        """
        初始化进化算法
        
        Args:
            config: 进化配置
            fitness_function: 适应度函数
        """
        self.config = config or EvolutionConfig()
        self.fitness_function = fitness_function or self._default_fitness_function
        
        self.population_size = self.config.population_size
        self.generation = 0
        self.agent_population: List[Agent] = []
        
        # 进化历史记录
        self.evolution_history: List[Dict[str, Any]] = []
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        
        # 停滞检测
        self.stagnation_counter = 0
        self.last_best_fitness = 0.0
        
    def initialize_population(
        self,
        genome_template: Dict[str, Any],
        initialization_strategy: str = 'random'
    ) -> List[Agent]:
        """
        初始化种群
        
        Args:
            genome_template: 基因组模板
            initialization_strategy: 初始化策略
            
        Returns:
            List[Agent]: 初始化的种群
        """
        self.agent_population = []
        
        for i in range(self.population_size):
            if initialization_strategy == 'random':
                genome = self._random_genome(genome_template)
            elif initialization_strategy == 'uniform':
                genome = copy.deepcopy(genome_template)
            else:
                genome = self._custom_initialization(genome_template)
                
            agent = Agent(
                agent_id=f"agent_{self.generation}_{i}",
                genome=genome,
                generation=self.generation
            )
            self.agent_population.append(agent)
            
        logger.info(f"Initialized population with {len(self.agent_population)} agents")
        return self.agent_population
        
    def evaluate_population_fitness(self) -> List[float]:
        """
        评估种群适应度
        
        Returns:
            List[float]: 适应度分数列表
        """
        fitness_scores = []
        
        for agent in self.agent_population:
            fitness = self.fitness_function(agent)
            agent.fitness = fitness
            fitness_scores.append(fitness)
            
        # 记录统计信息
        if fitness_scores:
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            max_fitness = max(fitness_scores)
            min_fitness = min(fitness_scores)
            
            self.avg_fitness_history.append(avg_fitness)
            self.best_fitness_history.append(max_fitness)
            
            logger.info(
                f"Generation {self.generation} fitness: "
                f"avg={avg_fitness:.4f}, max={max_fitness:.4f}, min={min_fitness:.4f}"
            )
            
        return fitness_scores
        
    def select_best_agents(
        self,
        fitness_scores: List[float],
        selection_method: str = 'tournament'
    ) -> List[Agent]:
        """
        选择优秀个体
        
        Args:
            fitness_scores: 适应度分数
            selection_method: 选择方法
            
        Returns:
            List[Agent]: 被选中的优秀个体
        """
        num_parents = int(self.population_size * (1 - self.config.elite_ratio))
        
        if selection_method == 'tournament':
            parents = self._tournament_selection(num_parents)
        elif selection_method == 'roulette':
            parents = self._roulette_selection(num_parents)
        elif selection_method == 'rank':
            parents = self._rank_selection(num_parents)
        else:
            parents = self._tournament_selection(num_parents)
            
        # 保留精英
        elite_count = int(self.population_size * self.config.elite_ratio)
        elites = self._select_elites(elite_count)
        
        logger.info(
            f"Selected {len(parents)} parents and {len(elites)} elites"
        )
        
        return parents + elites
        
    def crossover_and_mutate(
        self,
        parents: List[Agent]
    ) -> List[Agent]:
        """
        交叉和变异
        
        Args:
            parents: 父代个体
            
        Returns:
            List[Agent]: 子代个体
        """
        offspring = []
        
        # 保留精英（不进行交叉变异）
        elite_count = int(self.population_size * self.config.elite_ratio)
        elites = parents[-elite_count:] if elite_count > 0 else []
        breeding_parents = parents[:-elite_count] if elite_count > 0 else parents
        
        # 生成子代
        while len(offspring) + len(elites) < self.population_size:
            if len(breeding_parents) < 2:
                break
                
            # 选择两个父代
            parent1, parent2 = random.sample(breeding_parents, 2)
            
            # 交叉
            if random.random() < self.config.crossover_rate:
                child_genome = self._crossover(parent1.genome, parent2.genome)
            else:
                child_genome = copy.deepcopy(
                    parent1.genome if random.random() < 0.5 else parent2.genome
                )
                
            # 变异
            if random.random() < self.config.mutation_rate:
                child_genome = self._mutate(child_genome)
                
            # 创建子代个体
            child = Agent(
                agent_id=f"agent_{self.generation + 1}_{len(offspring)}",
                genome=child_genome,
                generation=self.generation + 1,
                parent_ids=[parent1.agent_id, parent2.agent_id]
            )
            offspring.append(child)
            
        # 将精英添加到子代（更新代数）
        for elite in elites:
            elite_copy = Agent(
                agent_id=f"agent_{self.generation + 1}_elite_{elite.agent_id}",
                genome=copy.deepcopy(elite.genome),
                fitness=elite.fitness,
                generation=self.generation + 1,
                parent_ids=[elite.agent_id]
            )
            offspring.append(elite_copy)
            
        logger.info(f"Generated {len(offspring)} offspring")
        return offspring
        
    def natural_selection(self, offspring: List[Agent]) -> None:
        """
        自然选择 - 用子代替换当前种群
        
        Args:
            offspring: 子代个体
        """
        # 记录进化历史
        best_agent = max(self.agent_population, key=lambda x: x.fitness)
        self.evolution_history.append({
            'generation': self.generation,
            'best_fitness': best_agent.fitness,
            'best_agent_id': best_agent.agent_id,
            'avg_fitness': sum(a.fitness for a in self.agent_population) / len(self.agent_population),
            'timestamp': datetime.now().isoformat()
        })
        
        # 检测停滞
        if best_agent.fitness <= self.last_best_fitness:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.last_best_fitness = best_agent.fitness
            
        # 用子代替换种群
        self.agent_population = offspring
        self.generation += 1
        
        logger.info(f"Natural selection completed. Generation: {self.generation}")
        
    def evolutionary_cycle(self) -> Dict[str, Any]:
        """
        完成一次进化循环
        
        Returns:
            Dict[str, Any]: 进化循环结果
        """
        # 1. 评估种群适应度
        fitness_scores = self.evaluate_population_fitness()
        
        # 2. 选择优秀个体
        parents = self.select_best_agents(fitness_scores)
        
        # 3. 交叉和变异
        offspring = self.crossover_and_mutate(parents)
        
        # 4. 环境选择
        self.natural_selection(offspring)
        
        # 返回本轮结果
        return {
            'generation': self.generation,
            'best_fitness': max(fitness_scores) if fitness_scores else 0,
            'avg_fitness': sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0,
            'stagnation_counter': self.stagnation_counter
        }
        
    def run_evolution(
        self,
        genome_template: Dict[str, Any],
        max_generations: Optional[int] = None
    ) -> Agent:
        """
        运行完整进化流程
        
        Args:
            genome_template: 基因组模板
            max_generations: 最大代数
            
        Returns:
            Agent: 最佳个体
        """
        max_gen = max_generations or self.config.max_generations
        
        # 初始化种群
        self.initialize_population(genome_template)
        
        # 进化循环
        for gen in range(max_gen):
            result = self.evolutionary_cycle()
            
            # 检查终止条件
            if result['best_fitness'] >= self.config.fitness_threshold:
                logger.info(f"Reached fitness threshold at generation {gen}")
                break
                
            if self.stagnation_counter >= self.config.stagnation_limit:
                logger.info(f"Stagnation detected at generation {gen}")
                # 可以触发多样性注入
                self._inject_diversity()
                self.stagnation_counter = 0
                
        # 返回最佳个体
        best_agent = max(self.agent_population, key=lambda x: x.fitness)
        logger.info(
            f"Evolution completed. Best fitness: {best_agent.fitness:.4f}"
        )
        
        return best_agent
        
    def _default_fitness_function(self, agent: Agent) -> float:
        """默认适应度函数"""
        # 基于基因组的简单适应度计算
        genome = agent.genome
        fitness = 0.0
        
        for key, value in genome.items():
            if isinstance(value, (int, float)):
                # 假设值越接近1越好
                fitness += 1.0 - abs(1.0 - value) / 10.0
                
        # 归一化
        if genome:
            fitness /= len(genome)
            
        return max(0.0, min(1.0, fitness))
        
    def _random_genome(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """生成随机基因组"""
        genome = {}
        
        for key, value in template.items():
            if isinstance(value, (int, float)):
                # 在原值附近随机扰动
                perturbation = random.uniform(-0.5, 0.5) * abs(value)
                genome[key] = value + perturbation
            elif isinstance(value, bool):
                genome[key] = random.random() > 0.5
            elif isinstance(value, str):
                genome[key] = value
            elif isinstance(value, list):
                genome[key] = copy.deepcopy(value)
            else:
                genome[key] = copy.deepcopy(value)
                
        return genome
        
    def _custom_initialization(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """自定义初始化"""
        return self._random_genome(template)
        
    def _tournament_selection(self, num_to_select: int) -> List[Agent]:
        """锦标赛选择"""
        selected = []
        
        for _ in range(num_to_select):
            tournament = random.sample(
                self.agent_population,
                min(self.config.tournament_size, len(self.agent_population))
            )
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
            
        return selected
        
    def _roulette_selection(self, num_to_select: int) -> List[Agent]:
        """轮盘赌选择"""
        total_fitness = sum(agent.fitness for agent in self.agent_population)
        if total_fitness == 0:
            return random.sample(self.agent_population, num_to_select)
            
        selected = []
        
        for _ in range(num_to_select):
            pick = random.uniform(0, total_fitness)
            current = 0
            for agent in self.agent_population:
                current += agent.fitness
                if current >= pick:
                    selected.append(agent)
                    break
                    
        return selected
        
    def _rank_selection(self, num_to_select: int) -> List[Agent]:
        """排名选择"""
        sorted_population = sorted(
            self.agent_population,
            key=lambda x: x.fitness,
            reverse=True
        )
        
        # 基于排名的概率分配
        n = len(sorted_population)
        ranks = list(range(n, 0, -1))
        total_rank = sum(ranks)
        probabilities = [r / total_rank for r in ranks]
        
        selected = []
        for _ in range(num_to_select):
            pick = random.random()
            current = 0
            for i, prob in enumerate(probabilities):
                current += prob
                if current >= pick:
                    selected.append(sorted_population[i])
                    break
                    
        return selected
        
    def _select_elites(self, elite_count: int) -> List[Agent]:
        """选择精英个体"""
        sorted_population = sorted(
            self.agent_population,
            key=lambda x: x.fitness,
            reverse=True
        )
        return sorted_population[:elite_count]
        
    def _crossover(
        self,
        genome1: Dict[str, Any],
        genome2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """基因组交叉"""
        child_genome = {}
        
        for key in genome1.keys():
            if key in genome2:
                # 均匀交叉
                if random.random() < 0.5:
                    child_genome[key] = copy.deepcopy(genome1[key])
                else:
                    child_genome[key] = copy.deepcopy(genome2[key])
                    
                # 对于数值类型，可以进行混合交叉
                if isinstance(genome1[key], (int, float)) and isinstance(genome2[key], (int, float)):
                    alpha = random.random()
                    child_genome[key] = alpha * genome1[key] + (1 - alpha) * genome2[key]
            else:
                child_genome[key] = copy.deepcopy(genome1[key])
                
        return child_genome
        
    def _mutate(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        """基因组变异"""
        mutated = copy.deepcopy(genome)
        
        for key, value in mutated.items():
            if random.random() < self.config.mutation_rate:
                if isinstance(value, (int, float)):
                    # 高斯变异
                    mutation = random.gauss(0, abs(value) * 0.1)
                    mutated[key] = value + mutation
                elif isinstance(value, bool):
                    mutated[key] = not value
                    
        return mutated
        
    def _inject_diversity(self) -> None:
        """注入多样性（当进化停滞时）"""
        # 替换部分个体为随机个体
        num_to_replace = int(self.population_size * 0.2)
        
        if self.agent_population:
            template = self.agent_population[0].genome
            
            # 按适应度排序，替换最差的个体
            sorted_population = sorted(
                self.agent_population,
                key=lambda x: x.fitness,
                reverse=True
            )
            
            for i in range(num_to_replace):
                if i < len(sorted_population):
                    new_genome = self._random_genome(template)
                    sorted_population[-(i+1)].genome = new_genome
                    sorted_population[-(i+1)].fitness = 0.0
                    
            self.agent_population = sorted_population
            
        logger.info(f"Injected diversity: replaced {num_to_replace} agents")
        
    def get_best_agent(self) -> Optional[Agent]:
        """获取当前最佳个体"""
        if not self.agent_population:
            return None
        return max(self.agent_population, key=lambda x: x.fitness)
        
    def get_population_stats(self) -> Dict[str, Any]:
        """获取种群统计信息"""
        if not self.agent_population:
            return {}
            
        fitnesses = [agent.fitness for agent in self.agent_population]
        
        return {
            'population_size': len(self.agent_population),
            'generation': self.generation,
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'max_fitness': max(fitnesses),
            'min_fitness': min(fitnesses),
            'fitness_std': self._calculate_std(fitnesses),
            'stagnation_counter': self.stagnation_counter
        }
        
    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
        
    def export_evolution_history(self) -> Dict[str, Any]:
        """导出进化历史"""
        return {
            'total_generations': self.generation,
            'evolution_history': self.evolution_history,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'final_best_agent': self.get_best_agent().to_dict() if self.get_best_agent() else None
        }
