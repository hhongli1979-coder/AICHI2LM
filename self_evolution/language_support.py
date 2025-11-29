"""
多语言支持 (Multi-Language Support)
===================================

实现大模型的多语言能力:
- 语言检测
- 多语言理解
- 多语言生成
- 语言翻译
- 跨语言推理
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re

logger = logging.getLogger(__name__)

# Cross-lingual relevance constants
BASE_CROSS_LINGUAL_RELEVANCE = 0.4
SAME_LANGUAGE_BONUS = 0.1


@dataclass
class LanguageProfile:
    """语言配置数据类"""
    language_code: str
    language_name: str
    native_name: str
    script: str  # 'latin', 'chinese', 'arabic', 'cyrillic', etc.
    direction: str = 'ltr'  # 'ltr' (left-to-right) or 'rtl' (right-to-left)
    supported_tasks: List[str] = field(default_factory=list)


@dataclass
class TranslationResult:
    """翻译结果数据类"""
    source_text: str
    source_language: str
    target_text: str
    target_language: str
    confidence: float
    alternative_translations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class LanguageDetectionResult:
    """语言检测结果数据类"""
    detected_language: str
    confidence: float
    alternative_languages: List[Tuple[str, float]] = field(default_factory=list)
    script_detected: str = ""


class MultiLanguageSupport:
    """
    多语言支持系统
    
    实现全面的多语言处理能力:
    1. 语言检测 - 自动识别输入文本的语言
    2. 多语言理解 - 理解不同语言的语义
    3. 多语言生成 - 生成指定语言的响应
    4. 语言翻译 - 跨语言翻译
    5. 跨语言推理 - 跨语言的知识推理
    """
    
    def __init__(self):
        """初始化多语言支持系统"""
        self.supported_languages = self._initialize_languages()
        self.language_patterns = self._initialize_patterns()
        self.translation_memory: Dict[str, TranslationResult] = {}
        self.language_statistics: Dict[str, int] = {}
        self.cross_lingual_mappings: Dict[str, Dict[str, str]] = {}
        
    def _initialize_languages(self) -> Dict[str, LanguageProfile]:
        """初始化支持的语言"""
        return {
            'zh': LanguageProfile(
                language_code='zh',
                language_name='Chinese',
                native_name='中文',
                script='chinese',
                supported_tasks=['understanding', 'generation', 'translation', 'reasoning']
            ),
            'en': LanguageProfile(
                language_code='en',
                language_name='English',
                native_name='English',
                script='latin',
                supported_tasks=['understanding', 'generation', 'translation', 'reasoning']
            ),
            'ja': LanguageProfile(
                language_code='ja',
                language_name='Japanese',
                native_name='日本語',
                script='japanese',
                supported_tasks=['understanding', 'generation', 'translation']
            ),
            'ko': LanguageProfile(
                language_code='ko',
                language_name='Korean',
                native_name='한국어',
                script='hangul',
                supported_tasks=['understanding', 'generation', 'translation']
            ),
            'fr': LanguageProfile(
                language_code='fr',
                language_name='French',
                native_name='Français',
                script='latin',
                supported_tasks=['understanding', 'generation', 'translation']
            ),
            'de': LanguageProfile(
                language_code='de',
                language_name='German',
                native_name='Deutsch',
                script='latin',
                supported_tasks=['understanding', 'generation', 'translation']
            ),
            'es': LanguageProfile(
                language_code='es',
                language_name='Spanish',
                native_name='Español',
                script='latin',
                supported_tasks=['understanding', 'generation', 'translation']
            ),
            'ru': LanguageProfile(
                language_code='ru',
                language_name='Russian',
                native_name='Русский',
                script='cyrillic',
                supported_tasks=['understanding', 'generation', 'translation']
            ),
            'ar': LanguageProfile(
                language_code='ar',
                language_name='Arabic',
                native_name='العربية',
                script='arabic',
                direction='rtl',
                supported_tasks=['understanding', 'generation', 'translation']
            ),
            'pt': LanguageProfile(
                language_code='pt',
                language_name='Portuguese',
                native_name='Português',
                script='latin',
                supported_tasks=['understanding', 'generation', 'translation']
            ),
        }
        
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """初始化语言检测模式"""
        return {
            'zh': [
                r'[\u4e00-\u9fff]',  # CJK统一汉字
                r'[\u3400-\u4dbf]',  # CJK扩展A
            ],
            'ja': [
                r'[\u3040-\u309f]',  # 平假名
                r'[\u30a0-\u30ff]',  # 片假名
            ],
            'ko': [
                r'[\uac00-\ud7af]',  # 韩文音节
                r'[\u1100-\u11ff]',  # 韩文字母
            ],
            'ar': [
                r'[\u0600-\u06ff]',  # 阿拉伯文
            ],
            'ru': [
                r'[\u0400-\u04ff]',  # 西里尔字母
            ],
            'en': [
                r'\b(the|is|are|was|were|have|has|had|do|does|did)\b',
                r'\b(and|or|but|if|then|else|when|where)\b',
            ],
            'fr': [
                r'\b(le|la|les|un|une|des|du|de|à|et|ou)\b',
                r'\b(est|sont|était|étaient|avoir|être)\b',
            ],
            'de': [
                r'\b(der|die|das|ein|eine|und|oder|ist|sind)\b',
                r'\b(haben|sein|werden|können|müssen)\b',
            ],
            'es': [
                r'\b(el|la|los|las|un|una|de|del|en|y|o)\b',
                r'\b(es|son|está|están|ser|estar)\b',
            ],
            'pt': [
                r'\b(o|a|os|as|um|uma|de|do|da|em|e|ou)\b',
                r'\b(é|são|está|estão|ser|estar)\b',
            ],
        }
        
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """
        检测文本语言
        
        Args:
            text: 输入文本
            
        Returns:
            LanguageDetectionResult: 语言检测结果
        """
        if not text or not text.strip():
            return LanguageDetectionResult(
                detected_language='unknown',
                confidence=0.0,
                script_detected='unknown'
            )
            
        scores: Dict[str, float] = {}
        script_detected = 'latin'  # 默认
        
        # 基于字符模式检测
        for lang_code, patterns in self.language_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
                
            if score > 0:
                scores[lang_code] = score
                
        # 检测脚本类型
        if re.search(r'[\u4e00-\u9fff]', text):
            script_detected = 'chinese'
            # 区分中文和日文（通过假名）
            if re.search(r'[\u3040-\u30ff]', text):
                script_detected = 'japanese'
        elif re.search(r'[\uac00-\ud7af]', text):
            script_detected = 'hangul'
        elif re.search(r'[\u0600-\u06ff]', text):
            script_detected = 'arabic'
        elif re.search(r'[\u0400-\u04ff]', text):
            script_detected = 'cyrillic'
            
        # 计算置信度并排序
        total_score = sum(scores.values()) if scores else 1
        normalized_scores = {
            lang: score / total_score
            for lang, score in scores.items()
        }
        
        if normalized_scores:
            detected_language = max(normalized_scores, key=normalized_scores.get)
            confidence = normalized_scores[detected_language]
            alternatives = sorted(
                [(lang, score) for lang, score in normalized_scores.items() if lang != detected_language],
                key=lambda x: x[1],
                reverse=True
            )[:3]
        else:
            # 默认为英语（拉丁字母）
            detected_language = 'en'
            confidence = 0.5
            alternatives = []
            
        # 更新统计
        self.language_statistics[detected_language] = \
            self.language_statistics.get(detected_language, 0) + 1
            
        result = LanguageDetectionResult(
            detected_language=detected_language,
            confidence=confidence,
            alternative_languages=alternatives,
            script_detected=script_detected
        )
        
        logger.info(
            f"Detected language: {detected_language} "
            f"(confidence={confidence:.2f}, script={script_detected})"
        )
        return result
        
    def understand(
        self,
        text: str,
        source_language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        多语言理解
        
        Args:
            text: 输入文本
            source_language: 源语言代码（可选，自动检测）
            
        Returns:
            Dict[str, Any]: 理解结果
        """
        # 检测语言（如果未指定）
        if source_language is None:
            detection = self.detect_language(text)
            source_language = detection.detected_language
            
        # 获取语言配置
        lang_profile = self.supported_languages.get(source_language)
        
        # 提取关键信息
        understanding = {
            'original_text': text,
            'language': source_language,
            'language_name': lang_profile.language_name if lang_profile else 'Unknown',
            'text_length': len(text),
            'word_count': self._count_words(text, source_language),
            'key_phrases': self._extract_key_phrases(text, source_language),
            'sentiment': self._analyze_sentiment(text, source_language),
            'topics': self._extract_topics(text, source_language),
            'intent': self._detect_intent(text, source_language),
            'entities': self._extract_entities(text, source_language)
        }
        
        logger.info(f"Understood text in {source_language}: {len(text)} chars")
        return understanding
        
    def generate(
        self,
        prompt: str,
        target_language: str,
        style: str = 'neutral',
        max_length: int = 1000
    ) -> Dict[str, Any]:
        """
        多语言生成
        
        Args:
            prompt: 生成提示
            target_language: 目标语言
            style: 风格 ('formal', 'casual', 'neutral')
            max_length: 最大长度
            
        Returns:
            Dict[str, Any]: 生成结果
        """
        lang_profile = self.supported_languages.get(target_language)
        
        if lang_profile is None:
            return {
                'success': False,
                'error': f'Unsupported language: {target_language}'
            }
            
        if 'generation' not in lang_profile.supported_tasks:
            return {
                'success': False,
                'error': f'Generation not supported for {target_language}'
            }
            
        # 生成响应
        generated_text = self._generate_text(prompt, target_language, style, max_length)
        
        result = {
            'success': True,
            'generated_text': generated_text,
            'target_language': target_language,
            'style': style,
            'length': len(generated_text)
        }
        
        logger.info(f"Generated {len(generated_text)} chars in {target_language}")
        return result
        
    def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> TranslationResult:
        """
        翻译文本
        
        Args:
            text: 源文本
            target_language: 目标语言
            source_language: 源语言（可选，自动检测）
            
        Returns:
            TranslationResult: 翻译结果
        """
        # 检测源语言（如果未指定）
        if source_language is None:
            detection = self.detect_language(text)
            source_language = detection.detected_language
            
        # 检查翻译记忆
        cache_key = f"{source_language}:{target_language}:{hash(text)}"
        if cache_key in self.translation_memory:
            cached = self.translation_memory[cache_key]
            logger.info("Retrieved translation from memory")
            return cached
            
        # 执行翻译
        translated_text = self._perform_translation(text, source_language, target_language)
        
        result = TranslationResult(
            source_text=text,
            source_language=source_language,
            target_text=translated_text,
            target_language=target_language,
            confidence=0.85,
            alternative_translations=[]
        )
        
        # 缓存翻译结果
        self.translation_memory[cache_key] = result
        
        logger.info(f"Translated from {source_language} to {target_language}")
        return result
        
    def cross_lingual_reasoning(
        self,
        query: str,
        knowledge_base: Dict[str, List[str]],
        target_language: str
    ) -> Dict[str, Any]:
        """
        跨语言推理
        
        Args:
            query: 查询
            knowledge_base: 多语言知识库 {language: [knowledge_items]}
            target_language: 目标语言
            
        Returns:
            Dict[str, Any]: 推理结果
        """
        # 检测查询语言
        query_detection = self.detect_language(query)
        query_language = query_detection.detected_language
        
        # 收集所有语言的相关知识
        relevant_knowledge = []
        
        for lang, knowledge_items in knowledge_base.items():
            for item in knowledge_items:
                # 计算相关性
                relevance = self._calculate_cross_lingual_relevance(query, item, query_language, lang)
                
                if relevance > 0.3:
                    relevant_knowledge.append({
                        'content': item,
                        'language': lang,
                        'relevance': relevance
                    })
                    
        # 按相关性排序
        relevant_knowledge.sort(key=lambda x: x['relevance'], reverse=True)
        
        # 整合知识并生成答案
        reasoning_result = self._synthesize_cross_lingual_answer(
            query,
            relevant_knowledge[:5],
            target_language
        )
        
        result = {
            'query': query,
            'query_language': query_language,
            'target_language': target_language,
            'relevant_knowledge': relevant_knowledge[:5],
            'reasoning': reasoning_result,
            'confidence': reasoning_result.get('confidence', 0.5)
        }
        
        logger.info(
            f"Cross-lingual reasoning: {query_language} -> {target_language}, "
            f"found {len(relevant_knowledge)} relevant items"
        )
        return result
        
    def get_language_profile(self, language_code: str) -> Optional[LanguageProfile]:
        """
        获取语言配置
        
        Args:
            language_code: 语言代码
            
        Returns:
            Optional[LanguageProfile]: 语言配置
        """
        return self.supported_languages.get(language_code)
        
    def list_supported_languages(self) -> List[Dict[str, str]]:
        """
        列出支持的语言
        
        Returns:
            List[Dict[str, str]]: 支持的语言列表
        """
        return [
            {
                'code': profile.language_code,
                'name': profile.language_name,
                'native_name': profile.native_name,
                'script': profile.script,
                'tasks': profile.supported_tasks
            }
            for profile in self.supported_languages.values()
        ]
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'supported_languages_count': len(self.supported_languages),
            'translation_memory_size': len(self.translation_memory),
            'language_usage': self.language_statistics,
            'most_used_language': max(
                self.language_statistics,
                key=self.language_statistics.get
            ) if self.language_statistics else None
        }
        
    def _count_words(self, text: str, language: str) -> int:
        """统计词数"""
        if language in ['zh', 'ja']:
            # 中文和日文按字符计数
            return len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff]', text))
        else:
            # 其他语言按空格分词
            return len(text.split())
            
    def _extract_key_phrases(self, text: str, language: str) -> List[str]:
        """提取关键短语"""
        # 简单实现：提取较长的词组
        words = text.split()
        phrases = []
        
        for i in range(len(words) - 1):
            phrase = ' '.join(words[i:i+2])
            if len(phrase) >= 5:
                phrases.append(phrase)
                
        return phrases[:5]
        
    def _analyze_sentiment(self, text: str, language: str) -> str:
        """分析情感"""
        # 简单的关键词情感分析
        positive_words = {
            'zh': ['好', '喜欢', '优秀', '棒', '满意'],
            'en': ['good', 'great', 'excellent', 'happy', 'love']
        }
        negative_words = {
            'zh': ['坏', '差', '糟糕', '讨厌', '失望'],
            'en': ['bad', 'terrible', 'awful', 'hate', 'disappointed']
        }
        
        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words.get(language, []) if w in text_lower)
        neg_count = sum(1 for w in negative_words.get(language, []) if w in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'
        
    def _extract_topics(self, text: str, language: str) -> List[str]:
        """提取主题"""
        # 简化实现
        return ['general']
        
    def _detect_intent(self, text: str, language: str) -> str:
        """检测意图"""
        # 简单的关键词意图检测
        question_patterns = {
            'zh': ['什么', '怎么', '为什么', '哪里', '谁', '？'],
            'en': ['what', 'how', 'why', 'where', 'who', '?']
        }
        
        text_lower = text.lower()
        patterns = question_patterns.get(language, question_patterns['en'])
        
        if any(p in text_lower for p in patterns):
            return 'question'
        return 'statement'
        
    def _extract_entities(self, text: str, language: str) -> List[Dict[str, str]]:
        """提取实体"""
        entities = []
        
        # 简单的命名实体识别（基于模式）
        # 数字
        numbers = re.findall(r'\d+', text)
        for num in numbers:
            entities.append({'type': 'NUMBER', 'value': num})
            
        return entities
        
    def _generate_text(
        self,
        prompt: str,
        target_language: str,
        style: str,
        max_length: int
    ) -> str:
        """生成文本"""
        # 这里是模拟实现，实际应该调用语言模型
        language_greetings = {
            'zh': '这是根据您的提示生成的中文响应。',
            'en': 'This is a response generated based on your prompt.',
            'ja': 'プロンプトに基づいて生成された応答です。',
            'ko': '프롬프트를 기반으로 생성된 응답입니다.',
            'fr': 'Ceci est une réponse générée en fonction de votre prompt.',
            'de': 'Dies ist eine Antwort, die auf Ihrer Eingabe basiert.',
            'es': 'Esta es una respuesta generada basada en su indicación.',
        }
        
        # Localized prompt labels
        prompt_labels = {
            'zh': '提示内容',
            'en': 'Prompt content',
            'ja': 'プロンプト内容',
            'ko': '프롬프트 내용',
            'fr': 'Contenu du prompt',
            'de': 'Eingabeinhalt',
            'es': 'Contenido del indicación',
        }
        
        base_response = language_greetings.get(
            target_language,
            language_greetings['en']
        )
        
        prompt_label = prompt_labels.get(
            target_language,
            prompt_labels['en']
        )
        
        return f"{base_response}\n\n{prompt_label}: {prompt[:100]}..."
        
    def _perform_translation(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """执行翻译"""
        # 这里是模拟实现，实际应该调用翻译服务
        translation_templates = {
            ('zh', 'en'): "[Translation from Chinese to English]: ",
            ('en', 'zh'): "[从英语翻译为中文]: ",
            ('ja', 'en'): "[Translation from Japanese to English]: ",
            ('ko', 'en'): "[Translation from Korean to English]: ",
        }
        
        prefix = translation_templates.get(
            (source_lang, target_lang),
            f"[Translation from {source_lang} to {target_lang}]: "
        )
        
        return f"{prefix}{text}"
        
    def _calculate_cross_lingual_relevance(
        self,
        query: str,
        item: str,
        query_lang: str,
        item_lang: str
    ) -> float:
        """计算跨语言相关性"""
        # 简化实现：基于文本长度和共同字符
        if query_lang == item_lang:
            # 同语言，直接计算词重叠
            query_words = set(query.lower().split())
            item_words = set(item.lower().split())
            overlap = len(query_words & item_words)
            return min(1.0, overlap / max(len(query_words), 1) + SAME_LANGUAGE_BONUS)
        else:
            # 不同语言，使用基础相关性分数
            return BASE_CROSS_LINGUAL_RELEVANCE
            
    def _synthesize_cross_lingual_answer(
        self,
        query: str,
        relevant_knowledge: List[Dict[str, Any]],
        target_language: str
    ) -> Dict[str, Any]:
        """整合跨语言答案"""
        if not relevant_knowledge:
            return {
                'answer': 'No relevant information found.',
                'confidence': 0.1,
                'sources': []
            }
            
        # 整合知识
        sources = [k['content'][:50] for k in relevant_knowledge]
        avg_relevance = sum(k['relevance'] for k in relevant_knowledge) / len(relevant_knowledge)
        
        return {
            'answer': f'Based on {len(relevant_knowledge)} knowledge sources across languages.',
            'confidence': avg_relevance,
            'sources': sources
        }
