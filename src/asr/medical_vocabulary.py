"""
Medical Vocabulary Manager
-------------------------
Manages medical hot words for speech recognition to improve accuracy
of medical terminology recognition in Chinese medical consultations.
"""

from typing import List, Dict, Any, Optional
import time
import logging

class MedicalVocabularyManager:
    """Manager for medical vocabulary/hot words used in speech recognition"""
    
    def __init__(self, target_model: str = "paraformer-realtime-8k-v2"):
        """Initialize medical vocabulary manager
        
        Args:
            target_model: Target ASR model for the vocabulary
        """
        self.target_model = target_model
        self.vocabulary_service = None
        self.vocabulary_id = None
        self._hotwords_cache = None
    
    def _ensure_api_key(self):
        """Ensure DashScope API key is properly set"""
        import dashscope
        import os
        from dotenv import load_dotenv
        import json
        
        # Check if API key is already set
        if hasattr(dashscope, 'api_key') and dashscope.api_key:
            return
        
        # Get the path to the .env file or config.json in the project root
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_path = os.path.join(root_dir, ".env")
        config_path = os.path.join(root_dir, "config.json")
        
        # Try to load from environment variable first
        if 'ALIBABA_API_KEY' in os.environ:
            dashscope.api_key = os.environ['ALIBABA_API_KEY']
            return
        
        if 'DASHSCOPE_API_KEY' in os.environ:
            dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
            return
        
        # Next try config.json
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'dashscope' in config and 'api_key' in config['dashscope']:
                        dashscope.api_key = config['dashscope']['api_key']
                        return
            except Exception as e:
                print(f"Error loading config.json: {e}")
        
        # Finally try .env file
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            if 'ALIBABA_API_KEY' in os.environ:
                dashscope.api_key = os.environ['ALIBABA_API_KEY']
                return
            if 'DASHSCOPE_API_KEY' in os.environ:
                dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
                return
        
    def create_vocabulary(self) -> Optional[str]:
        """Create medical vocabulary for better speech recognition
        
        Returns:
            Vocabulary ID if successful, None otherwise
        """
        try:
            import dashscope
            from dashscope.audio.asr.vocabulary import VocabularyService
            
            # Ensure API key is set
            self._ensure_api_key()
            
            # Get medical hot words
            medical_vocabulary = self.get_medical_hotwords()
            
            # Initialize vocabulary service
            self.vocabulary_service = VocabularyService()
            
            # Create vocabulary with a unique prefix (max 10 characters, letters and numbers only)
            timestamp = str(int(time.time()))[-6:]  # Use last 6 digits of timestamp
            prefix = f"med{timestamp}"
            
            try:
                self.vocabulary_id = self.vocabulary_service.create_vocabulary(
                    prefix=prefix,
                    target_model=self.target_model,
                    vocabulary=medical_vocabulary
                )
                print(f"Created medical vocabulary with ID: {self.vocabulary_id}")
                print(f"Added {len(medical_vocabulary)} medical hot words")
                return self.vocabulary_id
            except Exception as e:
                print(f"Failed to create medical vocabulary: {e}")
                print("Continuing without hot words...")
                return None
                
        except ImportError:
            print("Warning: dashscope.audio.asr.vocabulary not available")
            return None
        except Exception as e:
            print(f"Error setting up medical vocabulary: {e}")
            return None
    
    def get_medical_hotwords(self) -> List[Dict[str, Any]]:
        """Generate medical hot words for Chinese medical consultation
        
        Returns:
            List of medical hot words in the format required by DashScope
        """
        if self._hotwords_cache is not None:
            return self._hotwords_cache
            
        # Medical hot words categorized by consultation topics
        medical_terms = [
            # 基础信息 - Basic Information
            {"text": "男性", "weight": 5, "lang": "zh"},
            {"text": "女性", "weight": 5, "lang": "zh"},
            {"text": "慢性病", "weight": 5, "lang": "zh"},
            {"text": "高血压", "weight": 5, "lang": "zh"},
            {"text": "高血糖", "weight": 5, "lang": "zh"},
            {"text": "高血脂", "weight": 5, "lang": "zh"},
            {"text": "糖尿病", "weight": 5, "lang": "zh"},
            {"text": "胃病", "weight": 4, "lang": "zh"},
            {"text": "药物过敏", "weight": 5, "lang": "zh"},
            {"text": "过敏史", "weight": 4, "lang": "zh"},
            
            # 发热寒热 - Fever and Cold
            {"text": "发烧", "weight": 5, "lang": "zh"},
            {"text": "发热", "weight": 5, "lang": "zh"},
            {"text": "体温", "weight": 4, "lang": "zh"},
            {"text": "怕冷", "weight": 5, "lang": "zh"},
            {"text": "畏寒", "weight": 5, "lang": "zh"},
            {"text": "怯寒", "weight": 4, "lang": "zh"},
            {"text": "出汗", "weight": 4, "lang": "zh"},
            {"text": "盗汗", "weight": 4, "lang": "zh"},
            {"text": "自汗", "weight": 4, "lang": "zh"},
            {"text": "清水汗", "weight": 4, "lang": "zh"},
            {"text": "黏汗", "weight": 4, "lang": "zh"},
            
            # 头痛头晕 - Headache and Dizziness
            {"text": "头痛", "weight": 5, "lang": "zh"},
            {"text": "头疼", "weight": 5, "lang": "zh"},
            {"text": "头晕", "weight": 5, "lang": "zh"},
            {"text": "眩晕", "weight": 4, "lang": "zh"},
            {"text": "天旋地转", "weight": 4, "lang": "zh"},
            {"text": "头昏", "weight": 4, "lang": "zh"},
            {"text": "胀痛", "weight": 4, "lang": "zh"},
            {"text": "刺痛", "weight": 4, "lang": "zh"},
            {"text": "抽痛", "weight": 4, "lang": "zh"},
            {"text": "偏头痛", "weight": 4, "lang": "zh"},
            {"text": "恶心", "weight": 4, "lang": "zh"},
            {"text": "呕吐", "weight": 4, "lang": "zh"},
            
            # 五官 - Five Senses
            {"text": "眼干", "weight": 4, "lang": "zh"},
            {"text": "眼涩", "weight": 4, "lang": "zh"},
            {"text": "流泪", "weight": 4, "lang": "zh"},
            {"text": "视力模糊", "weight": 4, "lang": "zh"},
            {"text": "耳鸣", "weight": 4, "lang": "zh"},
            {"text": "听力下降", "weight": 4, "lang": "zh"},
            {"text": "鼻塞", "weight": 4, "lang": "zh"},
            {"text": "流鼻涕", "weight": 4, "lang": "zh"},
            {"text": "鼻涕", "weight": 4, "lang": "zh"},
            {"text": "打喷嚏", "weight": 4, "lang": "zh"},
            
            # 咽喉与咳嗽 - Throat and Cough
            {"text": "咽干", "weight": 4, "lang": "zh"},
            {"text": "咽痛", "weight": 4, "lang": "zh"},
            {"text": "喉咙痛", "weight": 4, "lang": "zh"},
            {"text": "喉咙干", "weight": 4, "lang": "zh"},
            {"text": "咳嗽", "weight": 5, "lang": "zh"},
            {"text": "干咳", "weight": 4, "lang": "zh"},
            {"text": "湿咳", "weight": 4, "lang": "zh"},
            {"text": "咳痰", "weight": 4, "lang": "zh"},
            {"text": "白痰", "weight": 4, "lang": "zh"},
            {"text": "黄痰", "weight": 4, "lang": "zh"},
            {"text": "浓痰", "weight": 4, "lang": "zh"},
            {"text": "稀痰", "weight": 4, "lang": "zh"},
            {"text": "胸闷", "weight": 4, "lang": "zh"},
            {"text": "心悸", "weight": 4, "lang": "zh"},
            {"text": "气短", "weight": 4, "lang": "zh"},
            
            # 食欲饮水 - Appetite and Hydration
            {"text": "食欲", "weight": 4, "lang": "zh"},
            {"text": "胃口", "weight": 4, "lang": "zh"},
            {"text": "口苦", "weight": 4, "lang": "zh"},
            {"text": "口干", "weight": 4, "lang": "zh"},
            {"text": "口渴", "weight": 4, "lang": "zh"},
            {"text": "反酸", "weight": 4, "lang": "zh"},
            {"text": "烧心", "weight": 4, "lang": "zh"},
            {"text": "嗳气", "weight": 4, "lang": "zh"},
            {"text": "饮水", "weight": 4, "lang": "zh"},
            {"text": "喝水", "weight": 4, "lang": "zh"},
            
            # 大小便 - Excretory System
            {"text": "小便", "weight": 4, "lang": "zh"},
            {"text": "尿液", "weight": 4, "lang": "zh"},
            {"text": "尿色", "weight": 4, "lang": "zh"},
            {"text": "尿黄", "weight": 4, "lang": "zh"},
            {"text": "尿频", "weight": 4, "lang": "zh"},
            {"text": "尿急", "weight": 4, "lang": "zh"},
            {"text": "尿痛", "weight": 4, "lang": "zh"},
            {"text": "大便", "weight": 4, "lang": "zh"},
            {"text": "便秘", "weight": 4, "lang": "zh"},
            {"text": "腹泻", "weight": 4, "lang": "zh"},
            {"text": "拉肚子", "weight": 4, "lang": "zh"},
            {"text": "腹痛", "weight": 4, "lang": "zh"},
            {"text": "肚子痛", "weight": 4, "lang": "zh"},
            {"text": "腹胀", "weight": 4, "lang": "zh"},
            {"text": "肚子胀", "weight": 4, "lang": "zh"},
            
            # 睡眠情绪 - Sleep and Mood
            {"text": "失眠", "weight": 4, "lang": "zh"},
            {"text": "睡眠", "weight": 4, "lang": "zh"},
            {"text": "入睡", "weight": 4, "lang": "zh"},
            {"text": "多梦", "weight": 4, "lang": "zh"},
            {"text": "早醒", "weight": 4, "lang": "zh"},
            {"text": "噩梦", "weight": 4, "lang": "zh"},
            {"text": "烦躁", "weight": 4, "lang": "zh"},
            {"text": "易怒", "weight": 4, "lang": "zh"},
            {"text": "抑郁", "weight": 4, "lang": "zh"},
            {"text": "焦虑", "weight": 4, "lang": "zh"},
            {"text": "情绪低落", "weight": 4, "lang": "zh"},
            
            # 皮肤症状 - Skin Symptoms
            {"text": "皮疹", "weight": 4, "lang": "zh"},
            {"text": "湿疹", "weight": 4, "lang": "zh"},
            {"text": "瘙痒", "weight": 4, "lang": "zh"},
            {"text": "皮肤痒", "weight": 4, "lang": "zh"},
            {"text": "红疹", "weight": 4, "lang": "zh"},
            {"text": "起疹子", "weight": 4, "lang": "zh"},
            
            # 女性症状 - Female Symptoms
            {"text": "月经", "weight": 4, "lang": "zh"},
            {"text": "例假", "weight": 4, "lang": "zh"},
            {"text": "大姨妈", "weight": 4, "lang": "zh"},
            {"text": "痛经", "weight": 4, "lang": "zh"},
            {"text": "白带", "weight": 4, "lang": "zh"},
            {"text": "月经周期", "weight": 4, "lang": "zh"},
            {"text": "月经量", "weight": 4, "lang": "zh"},
            {"text": "闭经", "weight": 4, "lang": "zh"},
            
            # 中医术语 - TCM Terms
            {"text": "气虚", "weight": 4, "lang": "zh"},
            {"text": "血虚", "weight": 4, "lang": "zh"},
            {"text": "阴虚", "weight": 4, "lang": "zh"},
            {"text": "阳虚", "weight": 4, "lang": "zh"},
            {"text": "湿热", "weight": 4, "lang": "zh"},
            {"text": "寒湿", "weight": 4, "lang": "zh"},
            {"text": "肝火", "weight": 4, "lang": "zh"},
            {"text": "肾虚", "weight": 4, "lang": "zh"},
            {"text": "脾虚", "weight": 4, "lang": "zh"},
            {"text": "心火", "weight": 4, "lang": "zh"},
            
            # 常用药物 - Common Medications
            {"text": "中药", "weight": 4, "lang": "zh"},
            {"text": "西药", "weight": 4, "lang": "zh"},
            {"text": "降压药", "weight": 4, "lang": "zh"},
            {"text": "降糖药", "weight": 4, "lang": "zh"},
            {"text": "感冒药", "weight": 4, "lang": "zh"},
            {"text": "止痛药", "weight": 4, "lang": "zh"},
            {"text": "消炎药", "weight": 4, "lang": "zh"},
            {"text": "抗生素", "weight": 4, "lang": "zh"},
            {"text": "激素", "weight": 4, "lang": "zh"},
            {"text": "维生素", "weight": 4, "lang": "zh"},
            
            # 常见药物过敏 - Common Drug Allergies
            {"text": "青霉素", "weight": 5, "lang": "zh"},
            {"text": "链霉素", "weight": 4, "lang": "zh"},
            {"text": "头孢", "weight": 5, "lang": "zh"},
            {"text": "头孢菌素", "weight": 4, "lang": "zh"},
            {"text": "阿莫西林", "weight": 4, "lang": "zh"},
            {"text": "氨苄西林", "weight": 4, "lang": "zh"},
            {"text": "红霉素", "weight": 4, "lang": "zh"},
            {"text": "磺胺", "weight": 4, "lang": "zh"},
            {"text": "磺胺类", "weight": 4, "lang": "zh"},
            {"text": "阿司匹林", "weight": 4, "lang": "zh"},
            {"text": "布洛芬", "weight": 4, "lang": "zh"},
            {"text": "对乙酰氨基酚", "weight": 4, "lang": "zh"},
            {"text": "扑热息痛", "weight": 4, "lang": "zh"},
            {"text": "庆大霉素", "weight": 4, "lang": "zh"},
            {"text": "卡那霉素", "weight": 4, "lang": "zh"},
            {"text": "氯霉素", "weight": 4, "lang": "zh"},
            {"text": "四环素", "weight": 4, "lang": "zh"},
            {"text": "土霉素", "weight": 4, "lang": "zh"},
            {"text": "金霉素", "weight": 4, "lang": "zh"},
            {"text": "利多卡因", "weight": 4, "lang": "zh"},
            {"text": "普鲁卡因", "weight": 4, "lang": "zh"},
            {"text": "碘", "weight": 4, "lang": "zh"},
            {"text": "碘伏", "weight": 4, "lang": "zh"},
            {"text": "碘酒", "weight": 4, "lang": "zh"},
            {"text": "造影剂", "weight": 4, "lang": "zh"},
            {"text": "胰岛素", "weight": 4, "lang": "zh"},
            
            # 常见过敏原 - Common Allergens
            {"text": "过敏", "weight": 5, "lang": "zh"},
            {"text": "过敏原", "weight": 4, "lang": "zh"},
            {"text": "过敏史", "weight": 4, "lang": "zh"},
            {"text": "药物过敏", "weight": 5, "lang": "zh"},
            {"text": "食物过敏", "weight": 4, "lang": "zh"},
            {"text": "花粉过敏", "weight": 4, "lang": "zh"},
            {"text": "尘螨", "weight": 4, "lang": "zh"},
            {"text": "尘螨过敏", "weight": 4, "lang": "zh"},
            {"text": "海鲜过敏", "weight": 4, "lang": "zh"},
            {"text": "鸡蛋过敏", "weight": 4, "lang": "zh"},
            {"text": "牛奶过敏", "weight": 4, "lang": "zh"},
            {"text": "坚果过敏", "weight": 4, "lang": "zh"},
            {"text": "花生过敏", "weight": 4, "lang": "zh"},
            {"text": "芒果过敏", "weight": 4, "lang": "zh"},
            {"text": "虾蟹过敏", "weight": 4, "lang": "zh"},
            {"text": "动物毛发", "weight": 4, "lang": "zh"},
            {"text": "猫毛过敏", "weight": 4, "lang": "zh"},
            {"text": "狗毛过敏", "weight": 4, "lang": "zh"},
            {"text": "霉菌", "weight": 4, "lang": "zh"},
            {"text": "霉菌过敏", "weight": 4, "lang": "zh"},
            {"text": "金属过敏", "weight": 4, "lang": "zh"},
            {"text": "镍过敏", "weight": 4, "lang": "zh"},
            {"text": "橡胶过敏", "weight": 4, "lang": "zh"},
            {"text": "乳胶过敏", "weight": 4, "lang": "zh"},
            
            # 过敏症状 - Allergic Symptoms
            {"text": "过敏反应", "weight": 5, "lang": "zh"},
            {"text": "过敏性休克", "weight": 5, "lang": "zh"},
            {"text": "荨麻疹", "weight": 5, "lang": "zh"},
            {"text": "风疹块", "weight": 4, "lang": "zh"},
            {"text": "皮肤红肿", "weight": 4, "lang": "zh"},
            {"text": "皮肤瘙痒", "weight": 4, "lang": "zh"},
            {"text": "过敏性鼻炎", "weight": 4, "lang": "zh"},
            {"text": "过敏性哮喘", "weight": 4, "lang": "zh"},
            {"text": "过敏性皮炎", "weight": 4, "lang": "zh"},
            {"text": "接触性皮炎", "weight": 4, "lang": "zh"},
            {"text": "湿疹过敏", "weight": 4, "lang": "zh"},
            {"text": "呼吸困难", "weight": 5, "lang": "zh"},
            {"text": "喉头水肿", "weight": 4, "lang": "zh"},
            {"text": "血管神经性水肿", "weight": 4, "lang": "zh"},
            
            # 时间相关 - Time Related
            {"text": "最近", "weight": 3, "lang": "zh"},
            {"text": "近期", "weight": 3, "lang": "zh"},
            {"text": "这几天", "weight": 3, "lang": "zh"},
            {"text": "这段时间", "weight": 3, "lang": "zh"},
            {"text": "平时", "weight": 3, "lang": "zh"},
            {"text": "经常", "weight": 3, "lang": "zh"},
            {"text": "偶尔", "weight": 3, "lang": "zh"},
            {"text": "有时候", "weight": 3, "lang": "zh"},
        ]
        
        self._hotwords_cache = medical_terms
        return medical_terms
    
    def cleanup_vocabulary(self):
        """Clean up the created vocabulary to avoid resource leakage"""
        if self.vocabulary_service and self.vocabulary_id:
            try:
                self.vocabulary_service.delete_vocabulary(self.vocabulary_id)
                print(f"Deleted medical vocabulary: {self.vocabulary_id}")
                self.vocabulary_id = None
            except Exception as e:
                print(f"Failed to delete vocabulary: {e}")
    
    def get_vocabulary_params(self) -> Dict[str, Any]:
        """Get vocabulary parameters for ASR configuration
        
        Returns:
            Dictionary with vocabulary parameters
        """
        if self.vocabulary_id:
            return {
                'vocabulary_id': self.vocabulary_id,
                'language_hints': ['zh']  # Required for vocabulary
            }
        return {}
    
    def query_vocabulary_info(self) -> Optional[Dict[str, Any]]:
        """Query vocabulary information
        
        Returns:
            Vocabulary information dictionary or None
        """
        if self.vocabulary_service and self.vocabulary_id:
            try:
                return self.vocabulary_service.query_vocabulary(self.vocabulary_id)
            except Exception as e:
                print(f"Failed to query vocabulary info: {e}")
        return None
    
    def get_vocabulary_statistics(self) -> Dict[str, Any]:
        """Get statistics about the medical vocabulary
        
        Returns:
            Dictionary with vocabulary statistics
        """
        hotwords = self.get_medical_hotwords()
        
        # Count by weight
        weight_counts = {}
        for word in hotwords:
            weight = word['weight']
            weight_counts[weight] = weight_counts.get(weight, 0) + 1
        
        # Count by categories (based on weights and content analysis)
        categories = {
            "核心症状词 (权重5)": len([w for w in hotwords if w['weight'] == 5]),
            "专业术语 (权重4)": len([w for w in hotwords if w['weight'] == 4]),
            "时间描述 (权重3)": len([w for w in hotwords if w['weight'] == 3])
        }
        
        return {
            "total_words": len(hotwords),
            "weight_distribution": weight_counts,
            "categories": categories,
            "vocabulary_id": self.vocabulary_id,
            "target_model": self.target_model
        }
    
    def __del__(self):
        """Destructor to clean up resources"""
        self.cleanup_vocabulary()


class MedicalEnhancedASR:
    """ASR wrapper that adds medical vocabulary support to base recognizers"""
    
    def __init__(self, base_recognizer, vocabulary_manager: Optional[MedicalVocabularyManager] = None):
        """Initialize medical-enhanced ASR
        
        Args:
            base_recognizer: Base ASR recognizer instance
            vocabulary_manager: Medical vocabulary manager (will create if None)
        """
        self.base_recognizer = base_recognizer
        self.vocabulary_manager = vocabulary_manager or MedicalVocabularyManager()
        self._vocab_initialized = False
        
        # Try to initialize vocabulary immediately for better user experience
        try:
            self._ensure_vocabulary_initialized()
        except Exception as e:
            print(f"Warning: Could not initialize vocabulary during construction: {e}")
            print("Vocabulary will be initialized on first use.")
        
    def _ensure_vocabulary_initialized(self):
        """Ensure medical vocabulary is initialized"""
        if not self._vocab_initialized:
            vocab_id = self.vocabulary_manager.create_vocabulary()
            self._vocab_initialized = True
            return vocab_id is not None
        return True
    
    def recognize_from_microphone(self) -> Dict[str, Any]:
        """Enhanced recognition with medical vocabulary support"""
        # Ensure vocabulary is initialized
        vocab_available = self._ensure_vocabulary_initialized()
        
        if vocab_available and hasattr(self.base_recognizer, 'recognize_with_vocabulary'):
            # Use vocabulary-enhanced recognition if available
            vocab_params = self.vocabulary_manager.get_vocabulary_params()
            result = self.base_recognizer.recognize_with_vocabulary(vocab_params)
        else:
            # Use base recognition
            result = self.base_recognizer.recognize_from_microphone()
            
        # Add vocabulary information to result
        if result.get("success") and vocab_available:
            result["medical_vocabulary_used"] = True
            result["vocabulary_id"] = self.vocabulary_manager.vocabulary_id
        
        return result
    
    def get_vocabulary_statistics(self) -> Dict[str, Any]:
        """Get medical vocabulary statistics"""
        return self.vocabulary_manager.get_vocabulary_statistics()
    
    def cleanup(self):
        """Clean up resources"""
        if self.vocabulary_manager:
            self.vocabulary_manager.cleanup_vocabulary()
        if hasattr(self.base_recognizer, 'cleanup'):
            self.base_recognizer.cleanup()
    
    def __getattr__(self, name):
        """Delegate other attributes to base recognizer"""
        return getattr(self.base_recognizer, name)
    
    def __del__(self):
        """Destructor"""
        self.cleanup()


 