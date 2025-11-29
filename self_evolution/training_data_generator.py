"""
自我训练数据生成 (Self Training Data Generator)
==============================================

实现自主训练数据的创建:
- 自我挑战生成
- 问题-解答对生成
- 自监督学习任务生成
- 数据质量验证
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import random
import json
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class TrainingPair:
    """训练对数据类"""
    question: str
    solution: str
    difficulty: float  # 0.0 - 1.0
    domain: str
    quality_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'question': self.question,
            'solution': self.solution,
            'difficulty': self.difficulty,
            'domain': self.domain,
            'quality_score': self.quality_score
        }


@dataclass
class SelfSupervisedTask:
    """自监督任务数据类"""
    task_type: str  # 'masked_prediction', 'next_sentence', 'contrastive', etc.
    input_text: str
    target: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentA:
    """
    问题生成智能体
    
    负责生成挑战性问题
    """
    
    def __init__(self, difficulty_range: Tuple[float, float] = (0.5, 1.0)):
        """
        初始化问题生成智能体
        
        Args:
            difficulty_range: 难度范围
        """
        self.difficulty_range = difficulty_range
        self.question_templates = self._load_question_templates()
        self.domains = [
            'mathematics', 'reasoning', 'coding', 'knowledge',
            'language', 'creativity', 'analysis', 'problem_solving'
        ]
        
    def _load_question_templates(self) -> Dict[str, List[str]]:
        """加载问题模板"""
        return {
            'mathematics': [
                '请解决以下数学问题：{problem}',
                '计算并解释：{expression}',
                '证明以下数学命题：{proposition}',
            ],
            'reasoning': [
                '分析以下情况并给出结论：{situation}',
                '根据给定条件推理：{conditions}',
                '找出以下逻辑谬误：{argument}',
            ],
            'coding': [
                '实现以下功能：{requirement}',
                '优化以下代码：{code}',
                '调试并修复：{buggy_code}',
            ],
            'knowledge': [
                '详细解释：{concept}',
                '比较和对比：{topics}',
                '分析{topic}的历史发展',
            ],
            'language': [
                '将以下文本翻译成{target_language}：{text}',
                '改写以下段落使其更{style}：{paragraph}',
                '总结以下文章的主要观点：{article}',
            ],
            'creativity': [
                '创作一个关于{theme}的故事',
                '设计一个解决{problem}的创新方案',
                '想象并描述{scenario}',
            ],
            'analysis': [
                '分析以下数据并得出结论：{data}',
                '评估{option_a}和{option_b}的优劣',
                '识别以下趋势的原因：{trend}',
            ],
            'problem_solving': [
                '解决以下实际问题：{problem}',
                '设计一个处理{challenge}的策略',
                '找出解决{issue}的最佳方法',
            ],
        }
        
    def generate_challenging_question(
        self,
        domain: Optional[str] = None,
        context: Optional[str] = None
    ) -> Tuple[str, str, float]:
        """
        生成挑战性问题
        
        Args:
            domain: 指定领域
            context: 上下文信息
            
        Returns:
            Tuple[str, str, float]: (问题, 领域, 难度)
        """
        if domain is None:
            domain = random.choice(self.domains)
            
        templates = self.question_templates.get(domain, self.question_templates['reasoning'])
        template = random.choice(templates)
        
        # 生成问题内容
        question_content = self._generate_question_content(domain, context)
        
        # 填充模板
        try:
            question = template.format(**question_content)
        except KeyError:
            question = f"[{domain}] {question_content.get('description', '请解答此问题')}"
            
        # 计算难度
        difficulty = random.uniform(*self.difficulty_range)
        
        return question, domain, difficulty
        
    def _generate_question_content(
        self,
        domain: str,
        context: Optional[str] = None
    ) -> Dict[str, str]:
        """生成问题内容"""
        # 基于领域生成相关内容
        content_generators = {
            'mathematics': lambda: {
                'problem': '求解方程 x^2 + 5x + 6 = 0',
                'expression': '积分 ∫x²dx',
                'proposition': '任意整数的平方是非负数',
                'description': '数学问题'
            },
            'reasoning': lambda: {
                'situation': '有三个人，每人说了一句话，只有一人说真话',
                'conditions': '如果A则B，如果B则C，已知C为假',
                'argument': '所有鸟都会飞，企鹅是鸟，所以企鹅会飞',
                'description': '逻辑推理问题'
            },
            'coding': lambda: {
                'requirement': '实现一个高效的字符串匹配算法',
                'code': 'def find_max(arr): return max(arr)',
                'buggy_code': 'for i in range(len(arr)): arr[i+1]',
                'description': '编程问题'
            },
            'knowledge': lambda: {
                'concept': '量子计算的基本原理',
                'topics': '机器学习和深度学习',
                'topic': '人工智能',
                'description': '知识问题'
            },
            'language': lambda: {
                'target_language': '英语',
                'text': '这是一段需要翻译的文本',
                'style': '正式',
                'paragraph': '这是一段需要改写的文字',
                'article': '这是一篇关于科技发展的文章',
                'description': '语言问题'
            },
            'creativity': lambda: {
                'theme': '未来城市',
                'problem': '环境污染',
                'scenario': '2050年的世界',
                'description': '创意问题'
            },
            'analysis': lambda: {
                'data': '过去十年的经济增长数据',
                'option_a': '方案A',
                'option_b': '方案B',
                'trend': '用户增长放缓',
                'description': '分析问题'
            },
            'problem_solving': lambda: {
                'problem': '如何提高团队协作效率',
                'challenge': '资源有限的情况下完成项目',
                'issue': '客户满意度下降',
                'description': '问题解决'
            },
        }
        
        generator = content_generators.get(domain, content_generators['reasoning'])
        content = generator()
        
        if context:
            content['context'] = context
            
        return content


class AgentB:
    """
    问题解答智能体
    
    负责解答AgentA生成的问题
    """
    
    def __init__(self):
        """初始化解答智能体"""
        self.solution_strategies = {
            'mathematics': self._solve_math,
            'reasoning': self._solve_reasoning,
            'coding': self._solve_coding,
            'knowledge': self._solve_knowledge,
            'language': self._solve_language,
            'creativity': self._solve_creativity,
            'analysis': self._solve_analysis,
            'problem_solving': self._solve_problem,
        }
        
    def solve_problem(
        self,
        question: str,
        domain: str
    ) -> str:
        """
        解答问题
        
        Args:
            question: 问题
            domain: 领域
            
        Returns:
            str: 解答
        """
        solver = self.solution_strategies.get(domain, self._solve_general)
        return solver(question)
        
    def _solve_math(self, question: str) -> str:
        """解答数学问题"""
        return f"""
解答：

让我来分析这个数学问题。

步骤1：理解问题
{question}

步骤2：应用相关数学原理
根据问题类型，我将使用适当的数学方法进行求解。

步骤3：详细计算过程
[计算过程详解]

步骤4：验证答案
通过代入验证或反向推导确认答案的正确性。

最终答案：[答案]

补充说明：这道题考查了[知识点]，在实际应用中有[应用场景]。
"""

    def _solve_reasoning(self, question: str) -> str:
        """解答推理问题"""
        return f"""
推理分析：

问题：{question}

分析步骤：

1. 识别关键信息
   - 提取问题中的核心条件
   - 明确已知信息和未知信息

2. 建立逻辑关系
   - 分析各条件之间的逻辑联系
   - 构建推理链条

3. 逐步推导
   - 从已知条件出发
   - 应用逻辑规则进行推导

4. 得出结论
   - 综合所有推理结果
   - 形成最终答案

结论：[结论]

推理过程说明：本题运用了[推理方法]，关键在于[关键点]。
"""

    def _solve_coding(self, question: str) -> str:
        """解答编程问题"""
        return f"""
编程解答：

问题：{question}

解决方案：

1. 问题分析
   - 输入输出要求
   - 边界条件处理
   - 时间空间复杂度要求

2. 算法设计
   - 选择合适的数据结构
   - 设计算法流程

3. 代码实现
```python
def solution(input_data):
    # 输入处理
    # 核心算法
    # 输出结果
    pass
```

4. 测试验证
   - 正常用例测试
   - 边界条件测试
   - 性能测试

复杂度分析：
- 时间复杂度：O(n)
- 空间复杂度：O(1)
"""

    def _solve_knowledge(self, question: str) -> str:
        """解答知识问题"""
        return f"""
知识解答：

问题：{question}

详细解释：

1. 概念定义
   [概念的准确定义]

2. 核心原理
   [相关的核心原理和机制]

3. 实际应用
   [在实际中的应用场景]

4. 发展历史
   [相关的发展历程]

5. 未来展望
   [可能的发展方向]

总结：[简要总结]
"""

    def _solve_language(self, question: str) -> str:
        """解答语言问题"""
        return f"""
语言任务解答：

任务：{question}

解答：

[翻译/改写/总结结果]

说明：
- 保持原文的核心意思
- 适应目标语言/风格的特点
- 确保表达准确流畅
"""

    def _solve_creativity(self, question: str) -> str:
        """解答创意问题"""
        return f"""
创意解答：

主题：{question}

创意内容：

[创意内容展示]

创意说明：
- 创意来源和灵感
- 核心创新点
- 实现可能性分析
"""

    def _solve_analysis(self, question: str) -> str:
        """解答分析问题"""
        return f"""
分析报告：

分析对象：{question}

1. 现状分析
   [当前状态描述]

2. 数据解读
   [关键数据的解读]

3. 原因分析
   [深层原因探究]

4. 趋势预测
   [未来发展趋势]

5. 建议方案
   [具体的建议和方案]

结论：[分析结论]
"""

    def _solve_problem(self, question: str) -> str:
        """解答问题解决类问题"""
        return f"""
问题解决方案：

问题描述：{question}

解决方案：

1. 问题定义
   - 明确问题的本质
   - 确定问题的边界

2. 原因分析
   - 分析问题产生的根本原因
   - 识别影响因素

3. 方案设计
   方案A：[方案描述]
   方案B：[方案描述]

4. 方案评估
   - 可行性分析
   - 成本效益分析

5. 实施建议
   - 具体实施步骤
   - 注意事项

推荐方案：[推荐的方案及理由]
"""

    def _solve_general(self, question: str) -> str:
        """通用解答"""
        return f"""
解答：

问题：{question}

分析与解答：

1. 问题理解
   [对问题的理解]

2. 解答思路
   [解答的思路和方法]

3. 详细解答
   [具体的解答内容]

4. 总结
   [简要总结]
"""


class SelfTrainingDataGenerator:
    """
    自我训练数据生成器
    
    实现自主创造训练数据:
    1. 自我挑战生成 - AgentA生成问题
    2. 问题解答 - AgentB解答问题
    3. 质量验证 - 验证解答质量
    4. 自监督学习 - 从对话历史中学习
    """
    
    def __init__(self):
        """初始化训练数据生成器"""
        self.agent_a = AgentA()
        self.agent_b = AgentB()
        self.generated_data: List[TrainingPair] = []
        self.conversation_history: List[Dict[str, Any]] = []
        self.quality_threshold = 0.6
        
    def create_training_data(
        self,
        num_samples: int = 1000,
        domains: Optional[List[str]] = None
    ) -> List[TrainingPair]:
        """
        自主创造训练数据
        
        Args:
            num_samples: 生成样本数量
            domains: 指定领域列表
            
        Returns:
            List[TrainingPair]: 训练数据对列表
        """
        training_pairs = []
        
        for i in range(num_samples):
            domain = None
            if domains:
                domain = random.choice(domains)
                
            # 智能体A生成问题
            question, actual_domain, difficulty = self.agent_a.generate_challenging_question(
                domain=domain
            )
            
            # 智能体B解答问题
            solution = self.agent_b.solve_problem(question, actual_domain)
            
            # 验证解答正确性
            quality_score = self.validate_solution(question, solution, actual_domain)
            
            if quality_score >= self.quality_threshold:
                pair = TrainingPair(
                    question=question,
                    solution=solution,
                    difficulty=difficulty,
                    domain=actual_domain,
                    quality_score=quality_score
                )
                training_pairs.append(pair)
                
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} training pairs")
                
        self.generated_data.extend(training_pairs)
        logger.info(
            f"Created {len(training_pairs)} training pairs "
            f"(filtered from {num_samples} attempts)"
        )
        
        return training_pairs
        
    def validate_solution(
        self,
        question: str,
        solution: str,
        domain: str
    ) -> float:
        """
        验证解答质量
        
        Args:
            question: 问题
            solution: 解答
            domain: 领域
            
        Returns:
            float: 质量分数 (0.0 - 1.0)
        """
        score = 0.0
        
        # 基本检查
        if len(solution) < 50:
            return 0.0
            
        if len(solution) > 100:
            score += 0.2
            
        # 结构检查
        if any(keyword in solution for keyword in ['步骤', '分析', '解答', '结论']):
            score += 0.3
            
        # 相关性检查
        question_keywords = set(question.lower().split())
        solution_keywords = set(solution.lower().split())
        overlap = len(question_keywords & solution_keywords)
        if overlap > 2:
            score += 0.2
            
        # 完整性检查
        if solution.strip().endswith(('。', '！', '）', ']', '```')):
            score += 0.2
            
        # 领域特定检查
        if domain == 'coding' and '```' in solution:
            score += 0.1
        elif domain == 'mathematics' and any(c in solution for c in ['=', '+', '-', '*', '/']):
            score += 0.1
        else:
            score += 0.1
            
        return min(1.0, score)
        
    def analyze_conversation_history(self) -> Dict[str, Any]:
        """
        分析对话历史
        
        Returns:
            Dict[str, Any]: 对话模式分析结果
        """
        if not self.conversation_history:
            return {}
            
        patterns = {
            'total_conversations': len(self.conversation_history),
            'domain_distribution': {},
            'avg_length': 0,
            'common_topics': [],
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        total_length = 0
        topic_counts = {}
        
        for conv in self.conversation_history:
            # 领域分布
            domain = conv.get('domain', 'unknown')
            patterns['domain_distribution'][domain] = (
                patterns['domain_distribution'].get(domain, 0) + 1
            )
            
            # 长度统计
            total_length += len(conv.get('content', ''))
            
            # 主题统计
            topics = conv.get('topics', [])
            for topic in topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
            # 质量分布
            quality = conv.get('quality', 0)
            if quality > 0.8:
                patterns['quality_distribution']['high'] += 1
            elif quality > 0.5:
                patterns['quality_distribution']['medium'] += 1
            else:
                patterns['quality_distribution']['low'] += 1
                
        if self.conversation_history:
            patterns['avg_length'] = total_length / len(self.conversation_history)
            
        # 获取最常见主题
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        patterns['common_topics'] = [topic for topic, _ in sorted_topics[:10]]
        
        return patterns
        
    def create_self_supervised_tasks(
        self,
        patterns: Dict[str, Any]
    ) -> List[SelfSupervisedTask]:
        """
        创建自监督训练任务
        
        Args:
            patterns: 对话模式分析结果
            
        Returns:
            List[SelfSupervisedTask]: 自监督任务列表
        """
        tasks = []
        
        # 基于已生成数据创建任务
        for pair in self.generated_data[-100:]:  # 使用最近100个样本
            # 掩码预测任务
            masked_text, target = self._create_masked_task(pair.solution)
            tasks.append(SelfSupervisedTask(
                task_type='masked_prediction',
                input_text=masked_text,
                target=target,
                metadata={'domain': pair.domain}
            ))
            
            # 下一句预测任务
            if len(pair.solution) > 200:
                first_half, second_half = self._split_text(pair.solution)
                tasks.append(SelfSupervisedTask(
                    task_type='next_sentence',
                    input_text=first_half,
                    target=second_half,
                    metadata={'domain': pair.domain}
                ))
                
            # 对比学习任务
            positive, negative = self._create_contrastive_task(pair)
            tasks.append(SelfSupervisedTask(
                task_type='contrastive',
                input_text=pair.question,
                target=positive,
                metadata={'negative': negative, 'domain': pair.domain}
            ))
            
        logger.info(f"Created {len(tasks)} self-supervised tasks")
        return tasks
        
    def self_supervised_learning(self) -> List[SelfSupervisedTask]:
        """
        执行自监督学习流程
        
        Returns:
            List[SelfSupervisedTask]: 自监督学习任务列表
        """
        # 从对话历史中提取学习模式
        conversation_patterns = self.analyze_conversation_history()
        
        # 生成自监督训练任务
        self_supervised_tasks = self.create_self_supervised_tasks(conversation_patterns)
        
        return self_supervised_tasks
        
    def train_on_self_generated_tasks(
        self,
        tasks: List[SelfSupervisedTask]
    ) -> Dict[str, Any]:
        """
        在自生成任务上训练
        
        Args:
            tasks: 自监督任务列表
            
        Returns:
            Dict[str, Any]: 训练结果统计
        """
        results = {
            'total_tasks': len(tasks),
            'task_types': {},
            'domains_covered': set()
        }
        
        for task in tasks:
            # 统计任务类型
            task_type = task.task_type
            results['task_types'][task_type] = results['task_types'].get(task_type, 0) + 1
            
            # 统计覆盖领域
            domain = task.metadata.get('domain')
            if domain:
                results['domains_covered'].add(domain)
                
        results['domains_covered'] = list(results['domains_covered'])
        
        logger.info(f"Training statistics: {results}")
        return results
        
    def _create_masked_task(self, text: str) -> Tuple[str, str]:
        """创建掩码预测任务"""
        words = text.split()
        if len(words) < 5:
            return text, ''
            
        mask_idx = random.randint(0, len(words) - 1)
        target = words[mask_idx]
        words[mask_idx] = '[MASK]'
        
        return ' '.join(words), target
        
    def _split_text(self, text: str) -> Tuple[str, str]:
        """分割文本"""
        mid = len(text) // 2
        # 找到最近的句子边界
        for i in range(mid, len(text)):
            if text[i] in '。！？':
                return text[:i+1], text[i+1:]
        return text[:mid], text[mid:]
        
    def _create_contrastive_task(self, pair: TrainingPair) -> Tuple[str, str]:
        """创建对比学习任务"""
        positive = pair.solution[:200] if len(pair.solution) > 200 else pair.solution
        
        # 创建负样本（从其他领域随机选择或打乱）
        other_pairs = [p for p in self.generated_data if p.domain != pair.domain]
        if other_pairs:
            negative_pair = random.choice(other_pairs)
            negative = negative_pair.solution[:200]
        else:
            # 打乱当前解答作为负样本
            words = pair.solution.split()
            random.shuffle(words)
            negative = ' '.join(words[:50])
            
        return positive, negative
        
    def add_conversation(self, conversation: Dict[str, Any]) -> None:
        """添加对话到历史"""
        self.conversation_history.append(conversation)
        
    def export_training_data(self, format: str = 'jsonl') -> List[Dict[str, Any]]:
        """
        导出训练数据
        
        Args:
            format: 导出格式
            
        Returns:
            List[Dict[str, Any]]: 导出的数据列表
        """
        return [pair.to_dict() for pair in self.generated_data]
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取生成统计信息"""
        if not self.generated_data:
            return {}
            
        domains = {}
        total_quality = 0
        
        for pair in self.generated_data:
            domains[pair.domain] = domains.get(pair.domain, 0) + 1
            total_quality += pair.quality_score
            
        return {
            'total_pairs': len(self.generated_data),
            'domain_distribution': domains,
            'avg_quality': total_quality / len(self.generated_data),
            'conversation_history_size': len(self.conversation_history)
        }
