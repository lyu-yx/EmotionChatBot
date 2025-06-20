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
        self._vocabulary_is_reused = False  # Flag to track if vocabulary is reused
    
    def _ensure_api_key(self) -> Optional[str]:
        """Ensure DashScope API key is properly set and return the API key
        
        Returns:
            API key string if found, None otherwise
        """
        import dashscope
        import os
        from dotenv import load_dotenv
        import json
        
        api_key = None
        
        # Check if API key is already set in dashscope
        if hasattr(dashscope, 'api_key') and dashscope.api_key:
            api_key = dashscope.api_key
        
        # Get the path to the .env file or config.json in the project root
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_path = os.path.join(root_dir, ".env")
        config_path = os.path.join(root_dir, "config.json")
        
        # Try to load from environment variable first
        if not api_key and 'ALIBABA_API_KEY' in os.environ:
            api_key = os.environ['ALIBABA_API_KEY']
        
        if not api_key and 'DASHSCOPE_API_KEY' in os.environ:
            api_key = os.environ['DASHSCOPE_API_KEY']
        
        # Next try config.json
        if not api_key and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'dashscope' in config and 'api_key' in config['dashscope']:
                        api_key = config['dashscope']['api_key']
            except Exception as e:
                print(f"Error loading config.json: {e}")
        
        # Finally try .env file
        if not api_key and os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            if 'ALIBABA_API_KEY' in os.environ:
                api_key = os.environ['ALIBABA_API_KEY']
            elif 'DASHSCOPE_API_KEY' in os.environ:
                api_key = os.environ['DASHSCOPE_API_KEY']
        
        # Set the global dashscope api_key for backward compatibility
        if api_key:
            dashscope.api_key = api_key
        
        return api_key
        
    def create_vocabulary(self) -> Optional[str]:
        """Create or reuse medical vocabulary for better speech recognition
        
        Returns:
            Vocabulary ID if successful, None otherwise
        """
        try:
            import dashscope
            from dashscope.audio.asr.vocabulary import VocabularyService
            
            # Ensure API key is set and get the API key
            api_key = self._ensure_api_key()
            if not api_key:
                print("Failed to get API key for vocabulary service")
                return None
            
            # Initialize vocabulary service with explicit API key
            self.vocabulary_service = VocabularyService(api_key=api_key)
            
            # First, check if there are existing vocabularies
            try:
                existing_vocabularies = self.vocabulary_service.list_vocabularies()
                print(f"Checking existing vocabularies: {existing_vocabularies}")
                
                # If there are existing vocabularies, use the first one
                if existing_vocabularies and len(existing_vocabularies) > 0:
                    first_vocab = existing_vocabularies[0]
                    if isinstance(first_vocab, dict) and 'vocabulary_id' in first_vocab:
                        self.vocabulary_id = first_vocab['vocabulary_id']
                        self._vocabulary_is_reused = True  # Mark as reused
                        vocab_status = first_vocab.get('status', 'Unknown')
                        create_time = first_vocab.get('gmt_create', 'Unknown')
                        print(f"✓ Reusing existing medical vocabulary: {self.vocabulary_id}")
                        print(f"  Status: {vocab_status}, Created: {create_time}")
                        return self.vocabulary_id
                    
            except Exception as e:
                print(f"Warning: Failed to list existing vocabularies: {e}")
                print("Will attempt to create a new vocabulary...")
            
            # If no existing vocabularies or failed to list, create a new one
            print("Creating new medical vocabulary...")
            
            # Get medical hot words
            medical_vocabulary = self.get_medical_hotwords()
            
            # Create vocabulary with a unique prefix (max 10 characters, letters and numbers only)
            timestamp = str(int(time.time()))[-6:]  # Use last 6 digits of timestamp
            prefix = f"med{timestamp}"
            
            try:
                self.vocabulary_id = self.vocabulary_service.create_vocabulary(
                    prefix=prefix,
                    target_model=self.target_model,
                    vocabulary=medical_vocabulary
                )
                self._vocabulary_is_reused = False  # Mark as newly created
                print(f"✓ Created new medical vocabulary with ID: {self.vocabulary_id}")
                print(f"  Added {len(medical_vocabulary)} medical hot words")
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
            # T1: 基础信息 - Basic Information
            {"text": "男性", "weight": 5, "lang": "zh"},
            {"text": "女性", "weight": 5, "lang": "zh"},
            {"text": "男", "weight": 5, "lang": "zh"},
            {"text": "女", "weight": 5, "lang": "zh"},
            {"text": "先生", "weight": 4, "lang": "zh"},
            {"text": "女士", "weight": 4, "lang": "zh"},
            {"text": "岁", "weight": 5, "lang": "zh"},
            {"text": "年龄", "weight": 5, "lang": "zh"},
            {"text": "慢性病", "weight": 5, "lang": "zh"},
            {"text": "高血压", "weight": 5, "lang": "zh"},
            {"text": "高血糖", "weight": 5, "lang": "zh"},
            {"text": "高血脂", "weight": 5, "lang": "zh"},
            {"text": "糖尿病", "weight": 5, "lang": "zh"},
            {"text": "胃病", "weight": 4, "lang": "zh"},
            {"text": "药物过敏", "weight": 5, "lang": "zh"},
            {"text": "过敏史", "weight": 4, "lang": "zh"},
            {"text": "治疗", "weight": 4, "lang": "zh"},
            {"text": "吃药", "weight": 4, "lang": "zh"},
            {"text": "服药", "weight": 4, "lang": "zh"},
            {"text": "正在治疗", "weight": 4, "lang": "zh"},
            
            # T2: 发热寒热 - Fever and Cold
            {"text": "发烧", "weight": 5, "lang": "zh"},
            {"text": "发热", "weight": 5, "lang": "zh"},
            {"text": "体温", "weight": 4, "lang": "zh"},
            {"text": "温度", "weight": 4, "lang": "zh"},
            {"text": "低烧", "weight": 4, "lang": "zh"},
            {"text": "高烧", "weight": 4, "lang": "zh"},
            {"text": "怕冷", "weight": 5, "lang": "zh"},
            {"text": "畏寒", "weight": 5, "lang": "zh"},
            {"text": "怯寒", "weight": 4, "lang": "zh"},
            {"text": "冷", "weight": 4, "lang": "zh"},
            {"text": "寒冷", "weight": 4, "lang": "zh"},
            {"text": "穿衣服", "weight": 4, "lang": "zh"},
            {"text": "盖被子", "weight": 4, "lang": "zh"},
            {"text": "缓解", "weight": 4, "lang": "zh"},
            {"text": "出汗", "weight": 4, "lang": "zh"},
            {"text": "盗汗", "weight": 4, "lang": "zh"},
            {"text": "自汗", "weight": 4, "lang": "zh"},
            {"text": "清水汗", "weight": 4, "lang": "zh"},
            {"text": "黏汗", "weight": 4, "lang": "zh"},
            {"text": "粘汗", "weight": 4, "lang": "zh"},
            {"text": "汗水", "weight": 4, "lang": "zh"},
            {"text": "流汗", "weight": 4, "lang": "zh"},
            {"text": "出汗多", "weight": 4, "lang": "zh"},
            {"text": "夜里出汗", "weight": 4, "lang": "zh"},
            {"text": "晚上出汗", "weight": 4, "lang": "zh"},
            {"text": "白天出汗", "weight": 4, "lang": "zh"},
            {"text": "早上", "weight": 3, "lang": "zh"},
            {"text": "中午", "weight": 3, "lang": "zh"},
            {"text": "下午", "weight": 3, "lang": "zh"},
            {"text": "晚上", "weight": 3, "lang": "zh"},
            {"text": "夜里", "weight": 3, "lang": "zh"},
            {"text": "时间段", "weight": 3, "lang": "zh"},
            
            # T3: 头痛头晕 - Headache and Dizziness
            {"text": "头痛", "weight": 5, "lang": "zh"},
            {"text": "头疼", "weight": 5, "lang": "zh"},
            {"text": "头晕", "weight": 5, "lang": "zh"},
            {"text": "眩晕", "weight": 4, "lang": "zh"},
            {"text": "天旋地转", "weight": 4, "lang": "zh"},
            {"text": "头昏", "weight": 4, "lang": "zh"},
            {"text": "昏沉", "weight": 4, "lang": "zh"},
            {"text": "头昏沉", "weight": 4, "lang": "zh"},
            {"text": "胀痛", "weight": 4, "lang": "zh"},
            {"text": "刺痛", "weight": 4, "lang": "zh"},
            {"text": "抽痛", "weight": 4, "lang": "zh"},
            {"text": "偏头痛", "weight": 4, "lang": "zh"},
            {"text": "头部", "weight": 4, "lang": "zh"},
            {"text": "部位", "weight": 4, "lang": "zh"},
            {"text": "前额", "weight": 4, "lang": "zh"},
            {"text": "后脑勺", "weight": 4, "lang": "zh"},
            {"text": "太阳穴", "weight": 4, "lang": "zh"},
            {"text": "头顶", "weight": 4, "lang": "zh"},
            {"text": "两侧", "weight": 4, "lang": "zh"},
            {"text": "左边", "weight": 4, "lang": "zh"},
            {"text": "右边", "weight": 4, "lang": "zh"},
            {"text": "恶心", "weight": 4, "lang": "zh"},
            {"text": "呕吐", "weight": 4, "lang": "zh"},
            {"text": "想吐", "weight": 4, "lang": "zh"},
            {"text": "恶心想吐", "weight": 4, "lang": "zh"},
            {"text": "伴随", "weight": 4, "lang": "zh"},
            
            # T4: 五官 - Five Senses
            {"text": "眼睛", "weight": 4, "lang": "zh"},
            {"text": "眼部", "weight": 4, "lang": "zh"},
            {"text": "眼干", "weight": 4, "lang": "zh"},
            {"text": "眼涩", "weight": 4, "lang": "zh"},
            {"text": "干涩", "weight": 4, "lang": "zh"},
            {"text": "发痒", "weight": 4, "lang": "zh"},
            {"text": "痒", "weight": 4, "lang": "zh"},
            {"text": "流泪", "weight": 4, "lang": "zh"},
            {"text": "视力", "weight": 4, "lang": "zh"},
            {"text": "视力模糊", "weight": 4, "lang": "zh"},
            {"text": "看不清", "weight": 4, "lang": "zh"},
            {"text": "模糊", "weight": 4, "lang": "zh"},
            {"text": "眼睛疼", "weight": 4, "lang": "zh"},
            {"text": "眼睛痛", "weight": 4, "lang": "zh"},
            {"text": "眼部不适", "weight": 4, "lang": "zh"},
            {"text": "耳朵", "weight": 4, "lang": "zh"},
            {"text": "耳部", "weight": 4, "lang": "zh"},
            {"text": "耳鸣", "weight": 4, "lang": "zh"},
            {"text": "听力", "weight": 4, "lang": "zh"},
            {"text": "听力下降", "weight": 4, "lang": "zh"},
            {"text": "听不清", "weight": 4, "lang": "zh"},
            {"text": "听力问题", "weight": 4, "lang": "zh"},
            {"text": "鼻子", "weight": 4, "lang": "zh"},
            {"text": "鼻部", "weight": 4, "lang": "zh"},
            {"text": "鼻塞", "weight": 4, "lang": "zh"},
            {"text": "流鼻涕", "weight": 4, "lang": "zh"},
            {"text": "鼻涕", "weight": 4, "lang": "zh"},
            {"text": "流涕", "weight": 4, "lang": "zh"},
            {"text": "打喷嚏", "weight": 4, "lang": "zh"},
            {"text": "鼻子堵", "weight": 4, "lang": "zh"},
            {"text": "不通气", "weight": 4, "lang": "zh"},
            
            # T5: 咽喉与咳嗽 - Throat and Cough
            {"text": "喉咙", "weight": 4, "lang": "zh"},
            {"text": "咽部", "weight": 4, "lang": "zh"},
            {"text": "嗓子", "weight": 4, "lang": "zh"},
            {"text": "咽干", "weight": 4, "lang": "zh"},
            {"text": "咽痛", "weight": 4, "lang": "zh"},
            {"text": "喉咙痛", "weight": 4, "lang": "zh"},
            {"text": "喉咙干", "weight": 4, "lang": "zh"},
            {"text": "嗓子疼", "weight": 4, "lang": "zh"},
            {"text": "嗓子干", "weight": 4, "lang": "zh"},
            {"text": "干", "weight": 4, "lang": "zh"},
            {"text": "痒", "weight": 4, "lang": "zh"},
            {"text": "疼", "weight": 4, "lang": "zh"},
            {"text": "痛", "weight": 4, "lang": "zh"},
            {"text": "堵", "weight": 4, "lang": "zh"},
            {"text": "不适", "weight": 4, "lang": "zh"},
            {"text": "咳嗽", "weight": 5, "lang": "zh"},
            {"text": "咳", "weight": 5, "lang": "zh"},
            {"text": "干咳", "weight": 4, "lang": "zh"},
            {"text": "湿咳", "weight": 4, "lang": "zh"},
            {"text": "间断", "weight": 4, "lang": "zh"},
            {"text": "持续", "weight": 4, "lang": "zh"},
            {"text": "一直咳", "weight": 4, "lang": "zh"},
            {"text": "偶尔咳", "weight": 4, "lang": "zh"},
            {"text": "有时咳", "weight": 4, "lang": "zh"},
            {"text": "咳痰", "weight": 4, "lang": "zh"},
            {"text": "有痰", "weight": 4, "lang": "zh"},
            {"text": "白痰", "weight": 4, "lang": "zh"},
            {"text": "黄痰", "weight": 4, "lang": "zh"},
            {"text": "绿痰", "weight": 4, "lang": "zh"},
            {"text": "浓痰", "weight": 4, "lang": "zh"},
            {"text": "稀痰", "weight": 4, "lang": "zh"},
            {"text": "粘痰", "weight": 4, "lang": "zh"},
            {"text": "痰多", "weight": 4, "lang": "zh"},
            {"text": "痰少", "weight": 4, "lang": "zh"},
            {"text": "容易咳出", "weight": 4, "lang": "zh"},
            {"text": "不容易咳出", "weight": 4, "lang": "zh"},
            {"text": "咳不出", "weight": 4, "lang": "zh"},
            {"text": "难咳出", "weight": 4, "lang": "zh"},
            {"text": "胸闷", "weight": 4, "lang": "zh"},
            {"text": "心悸", "weight": 4, "lang": "zh"},
            {"text": "气短", "weight": 4, "lang": "zh"},
            {"text": "憋气", "weight": 4, "lang": "zh"},
            {"text": "透不过气", "weight": 4, "lang": "zh"},
            {"text": "心跳快", "weight": 4, "lang": "zh"},
            {"text": "心慌", "weight": 4, "lang": "zh"},
            
            # T6: 食欲饮水 - Appetite and Hydration
            {"text": "食欲", "weight": 4, "lang": "zh"},
            {"text": "胃口", "weight": 4, "lang": "zh"},
            {"text": "食欲好", "weight": 4, "lang": "zh"},
            {"text": "食欲差", "weight": 4, "lang": "zh"},
            {"text": "胃口好", "weight": 4, "lang": "zh"},
            {"text": "胃口差", "weight": 4, "lang": "zh"},
            {"text": "不想吃", "weight": 4, "lang": "zh"},
            {"text": "吃得少", "weight": 4, "lang": "zh"},
            {"text": "吃得多", "weight": 4, "lang": "zh"},
            {"text": "偏好", "weight": 4, "lang": "zh"},
            {"text": "喜欢", "weight": 4, "lang": "zh"},
            {"text": "冷食", "weight": 4, "lang": "zh"},
            {"text": "热食", "weight": 4, "lang": "zh"},
            {"text": "凉的", "weight": 4, "lang": "zh"},
            {"text": "热的", "weight": 4, "lang": "zh"},
            {"text": "温的", "weight": 4, "lang": "zh"},
            {"text": "口腔", "weight": 4, "lang": "zh"},
            {"text": "口苦", "weight": 4, "lang": "zh"},
            {"text": "口干", "weight": 4, "lang": "zh"},
            {"text": "口渴", "weight": 4, "lang": "zh"},
            {"text": "反酸", "weight": 4, "lang": "zh"},
            {"text": "烧心", "weight": 4, "lang": "zh"},
            {"text": "嗳气", "weight": 4, "lang": "zh"},
            {"text": "打嗝", "weight": 4, "lang": "zh"},
            {"text": "饮水", "weight": 4, "lang": "zh"},
            {"text": "喝水", "weight": 4, "lang": "zh"},
            {"text": "喝水习惯", "weight": 4, "lang": "zh"},
            {"text": "饮水习惯", "weight": 4, "lang": "zh"},
            {"text": "热水", "weight": 4, "lang": "zh"},
            {"text": "冷水", "weight": 4, "lang": "zh"},
            {"text": "温水", "weight": 4, "lang": "zh"},
            {"text": "常温水", "weight": 4, "lang": "zh"},
            {"text": "多喝水", "weight": 4, "lang": "zh"},
            {"text": "少喝水", "weight": 4, "lang": "zh"},
            
            # T7: 大小便与腹痛 - Excretory System and Abdominal Pain
            {"text": "小便", "weight": 4, "lang": "zh"},
            {"text": "尿", "weight": 4, "lang": "zh"},
            {"text": "尿液", "weight": 4, "lang": "zh"},
            {"text": "通畅", "weight": 4, "lang": "zh"},
            {"text": "小便通畅", "weight": 4, "lang": "zh"},
            {"text": "尿色", "weight": 4, "lang": "zh"},
            {"text": "颜色", "weight": 4, "lang": "zh"},
            {"text": "尿黄", "weight": 4, "lang": "zh"},
            {"text": "尿白", "weight": 4, "lang": "zh"},
            {"text": "尿红", "weight": 4, "lang": "zh"},
            {"text": "黄色", "weight": 4, "lang": "zh"},
            {"text": "红色", "weight": 4, "lang": "zh"},
            {"text": "清澈", "weight": 4, "lang": "zh"},
            {"text": "浑浊", "weight": 4, "lang": "zh"},
            {"text": "尿频", "weight": 4, "lang": "zh"},
            {"text": "尿急", "weight": 4, "lang": "zh"},
            {"text": "尿痛", "weight": 4, "lang": "zh"},
            {"text": "大便", "weight": 4, "lang": "zh"},
            {"text": "拉屎", "weight": 4, "lang": "zh"},
            {"text": "排便", "weight": 4, "lang": "zh"},
            {"text": "大便情况", "weight": 4, "lang": "zh"},
            {"text": "次数", "weight": 4, "lang": "zh"},
            {"text": "形状", "weight": 4, "lang": "zh"},
            {"text": "正常", "weight": 4, "lang": "zh"},
            {"text": "便秘", "weight": 4, "lang": "zh"},
            {"text": "腹泻", "weight": 4, "lang": "zh"},
            {"text": "拉肚子", "weight": 4, "lang": "zh"},
            {"text": "干燥", "weight": 4, "lang": "zh"},
            {"text": "稀", "weight": 4, "lang": "zh"},
            {"text": "成形", "weight": 4, "lang": "zh"},
            {"text": "不成形", "weight": 4, "lang": "zh"},
            {"text": "一天几次", "weight": 4, "lang": "zh"},
            {"text": "几天一次", "weight": 4, "lang": "zh"},
            {"text": "腹痛", "weight": 4, "lang": "zh"},
            {"text": "肚子痛", "weight": 4, "lang": "zh"},
            {"text": "腹胀", "weight": 4, "lang": "zh"},
            {"text": "肚子胀", "weight": 4, "lang": "zh"},
            {"text": "位置", "weight": 4, "lang": "zh"},
            {"text": "上腹", "weight": 4, "lang": "zh"},
            {"text": "下腹", "weight": 4, "lang": "zh"},
            {"text": "左腹", "weight": 4, "lang": "zh"},
            {"text": "右腹", "weight": 4, "lang": "zh"},
            {"text": "肚脐", "weight": 4, "lang": "zh"},
            {"text": "排便后", "weight": 4, "lang": "zh"},
            {"text": "缓解", "weight": 4, "lang": "zh"},
            {"text": "不缓解", "weight": 4, "lang": "zh"},
            
            # T8: 睡眠 - Sleep
            {"text": "睡眠", "weight": 4, "lang": "zh"},
            {"text": "睡觉", "weight": 4, "lang": "zh"},
            {"text": "睡眠质量", "weight": 4, "lang": "zh"},
            {"text": "睡得好", "weight": 4, "lang": "zh"},
            {"text": "睡得不好", "weight": 4, "lang": "zh"},
            {"text": "良好", "weight": 4, "lang": "zh"},
            {"text": "不好", "weight": 4, "lang": "zh"},
            {"text": "失眠", "weight": 4, "lang": "zh"},
            {"text": "入睡", "weight": 4, "lang": "zh"},
            {"text": "容易入睡", "weight": 4, "lang": "zh"},
            {"text": "不容易入睡", "weight": 4, "lang": "zh"},
            {"text": "难入睡", "weight": 4, "lang": "zh"},
            {"text": "睡不着", "weight": 4, "lang": "zh"},
            {"text": "多梦", "weight": 4, "lang": "zh"},
            {"text": "早醒", "weight": 4, "lang": "zh"},
            {"text": "噩梦", "weight": 4, "lang": "zh"},
            {"text": "做梦", "weight": 4, "lang": "zh"},
            {"text": "梦多", "weight": 4, "lang": "zh"},
            {"text": "梦少", "weight": 4, "lang": "zh"},
            {"text": "睡得浅", "weight": 4, "lang": "zh"},
            {"text": "睡得深", "weight": 4, "lang": "zh"},
            {"text": "易醒", "weight": 4, "lang": "zh"},
            {"text": "半夜醒", "weight": 4, "lang": "zh"},
            
            # T8.1: 情绪 - Mood and Emotion
            {"text": "情绪", "weight": 4, "lang": "zh"},
            {"text": "情绪状态", "weight": 4, "lang": "zh"},
            {"text": "烦躁", "weight": 4, "lang": "zh"},
            {"text": "发怒", "weight": 4, "lang": "zh"},
            {"text": "易怒", "weight": 4, "lang": "zh"},
            {"text": "闷闷不乐", "weight": 4, "lang": "zh"},
            {"text": "不开心", "weight": 4, "lang": "zh"},
            {"text": "郁闷", "weight": 4, "lang": "zh"},
            {"text": "抑郁", "weight": 4, "lang": "zh"},
            {"text": "焦虑", "weight": 4, "lang": "zh"},
            {"text": "紧张", "weight": 4, "lang": "zh"},
            {"text": "情绪低落", "weight": 4, "lang": "zh"},
            {"text": "心情好", "weight": 4, "lang": "zh"},
            {"text": "心情不好", "weight": 4, "lang": "zh"},
            {"text": "开心", "weight": 4, "lang": "zh"},
            {"text": "高兴", "weight": 4, "lang": "zh"},
            {"text": "愉快", "weight": 4, "lang": "zh"},
            {"text": "平静", "weight": 4, "lang": "zh"},
            
            # T8.2: 皮肤 - Skin Symptoms
            {"text": "皮肤", "weight": 4, "lang": "zh"},
            {"text": "皮肤症状", "weight": 4, "lang": "zh"},
            {"text": "异常", "weight": 4, "lang": "zh"},
            {"text": "皮疹", "weight": 4, "lang": "zh"},
            {"text": "湿疹", "weight": 4, "lang": "zh"},
            {"text": "瘙痒", "weight": 4, "lang": "zh"},
            {"text": "皮肤痒", "weight": 4, "lang": "zh"},
            {"text": "痒", "weight": 4, "lang": "zh"},
            {"text": "发痒", "weight": 4, "lang": "zh"},
            {"text": "红疹", "weight": 4, "lang": "zh"},
            {"text": "起疹子", "weight": 4, "lang": "zh"},
            {"text": "疹子", "weight": 4, "lang": "zh"},
            {"text": "起红点", "weight": 4, "lang": "zh"},
            {"text": "红点", "weight": 4, "lang": "zh"},
            {"text": "红斑", "weight": 4, "lang": "zh"},
            {"text": "皮肤红", "weight": 4, "lang": "zh"},
            {"text": "发红", "weight": 4, "lang": "zh"},
            {"text": "脱皮", "weight": 4, "lang": "zh"},
            {"text": "干燥", "weight": 4, "lang": "zh"},
            {"text": "皮肤干", "weight": 4, "lang": "zh"},
            {"text": "粗糙", "weight": 4, "lang": "zh"},
            {"text": "光滑", "weight": 4, "lang": "zh"},
            {"text": "正常", "weight": 4, "lang": "zh"},
            {"text": "没有异常", "weight": 4, "lang": "zh"},
            
            # T9: 女性月经 - Female Symptoms
            {"text": "月经", "weight": 4, "lang": "zh"},
            {"text": "例假", "weight": 4, "lang": "zh"},
            {"text": "大姨妈", "weight": 4, "lang": "zh"},
            {"text": "生理期", "weight": 4, "lang": "zh"},
            {"text": "月经周期", "weight": 4, "lang": "zh"},
            {"text": "周期", "weight": 4, "lang": "zh"},
            {"text": "规律", "weight": 4, "lang": "zh"},
            {"text": "不规律", "weight": 4, "lang": "zh"},
            {"text": "正常", "weight": 4, "lang": "zh"},
            {"text": "异常", "weight": 4, "lang": "zh"},
            {"text": "颜色", "weight": 4, "lang": "zh"},
            {"text": "量", "weight": 4, "lang": "zh"},
            {"text": "月经量", "weight": 4, "lang": "zh"},
            {"text": "多", "weight": 4, "lang": "zh"},
            {"text": "少", "weight": 4, "lang": "zh"},
            {"text": "量多", "weight": 4, "lang": "zh"},
            {"text": "量少", "weight": 4, "lang": "zh"},
            {"text": "红色", "weight": 4, "lang": "zh"},
            {"text": "暗红", "weight": 4, "lang": "zh"},
            {"text": "鲜红", "weight": 4, "lang": "zh"},
            {"text": "黑色", "weight": 4, "lang": "zh"},
            {"text": "褐色", "weight": 4, "lang": "zh"},
            {"text": "痛经", "weight": 4, "lang": "zh"},
            {"text": "肚子疼", "weight": 4, "lang": "zh"},
            {"text": "腹痛", "weight": 4, "lang": "zh"},
            {"text": "疼痛", "weight": 4, "lang": "zh"},
            {"text": "白带", "weight": 4, "lang": "zh"},
            {"text": "白带情况", "weight": 4, "lang": "zh"},
            {"text": "分泌物", "weight": 4, "lang": "zh"},
            {"text": "白色", "weight": 4, "lang": "zh"},
            {"text": "黄色", "weight": 4, "lang": "zh"},
            {"text": "透明", "weight": 4, "lang": "zh"},
            {"text": "异味", "weight": 4, "lang": "zh"},
            {"text": "无味", "weight": 4, "lang": "zh"},
            {"text": "闭经", "weight": 4, "lang": "zh"},
            {"text": "推迟", "weight": 4, "lang": "zh"},
            {"text": "提前", "weight": 4, "lang": "zh"},
            
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
            
            # 常用否定与确认词 - Common Negative and Affirmative Words
            {"text": "没有", "weight": 5, "lang": "zh"},
            {"text": "的确", "weight": 4, "lang": "zh"},
            {"text": "还好", "weight": 4, "lang": "zh"},
            {"text": "一般", "weight": 4, "lang": "zh"},
            {"text": "还行", "weight": 4, "lang": "zh"},
            {"text": "可以", "weight": 4, "lang": "zh"},
            {"text": "不可以", "weight": 4, "lang": "zh"},
            {"text": "行", "weight": 4, "lang": "zh"},
            {"text": "不行", "weight": 4, "lang": "zh"},
            
            # 时间相关 - Time Related
            {"text": "最近", "weight": 3, "lang": "zh"},
            {"text": "近期", "weight": 3, "lang": "zh"},
            {"text": "这几天", "weight": 3, "lang": "zh"},
            {"text": "这段时间", "weight": 3, "lang": "zh"},
            {"text": "平时", "weight": 3, "lang": "zh"},
            {"text": "经常", "weight": 3, "lang": "zh"},
            {"text": "偶尔", "weight": 3, "lang": "zh"},
            {"text": "有时候", "weight": 3, "lang": "zh"},
            {"text": "总是", "weight": 3, "lang": "zh"},
            {"text": "从来不", "weight": 3, "lang": "zh"},
            {"text": "以前", "weight": 3, "lang": "zh"},
            {"text": "现在", "weight": 3, "lang": "zh"},
            {"text": "之前", "weight": 3, "lang": "zh"},
            {"text": "以后", "weight": 3, "lang": "zh"},

            # 程度副词 - Degree Adverbs
            {"text": "非常", "weight": 4, "lang": "zh"},
            {"text": "特别", "weight": 4, "lang": "zh"},
            {"text": "比较", "weight": 4, "lang": "zh"},
            {"text": "有点", "weight": 4, "lang": "zh"},
            {"text": "有些", "weight": 4, "lang": "zh"},
            {"text": "稍微", "weight": 4, "lang": "zh"},
            {"text": "轻微", "weight": 4, "lang": "zh"},
            {"text": "严重", "weight": 4, "lang": "zh"},
            {"text": "厉害", "weight": 4, "lang": "zh"},
            {"text": "明显", "weight": 4, "lang": "zh"},
            {"text": "不明显", "weight": 4, "lang": "zh"},
            {"text": "强烈", "weight": 4, "lang": "zh"},
            {"text": "激烈", "weight": 4, "lang": "zh"},
            {"text": "剧烈", "weight": 4, "lang": "zh"},
            {"text": "轻度", "weight": 4, "lang": "zh"},
            {"text": "中度", "weight": 4, "lang": "zh"},
            {"text": "重度", "weight": 4, "lang": "zh"},
            {"text": "极度", "weight": 4, "lang": "zh"},
            
            # 口语化表达 - Colloquial Expressions
            {"text": "挺", "weight": 4, "lang": "zh"},
            {"text": "蛮", "weight": 4, "lang": "zh"},
            {"text": "还蛮", "weight": 4, "lang": "zh"},
            {"text": "挺严重", "weight": 4, "lang": "zh"},
            {"text": "不太", "weight": 4, "lang": "zh"},
            {"text": "不怎么", "weight": 4, "lang": "zh"},
            {"text": "还可以", "weight": 4, "lang": "zh"},
            {"text": "马马虎虎", "weight": 4, "lang": "zh"},
            {"text": "凑合", "weight": 4, "lang": "zh"},
            {"text": "还不错", "weight": 4, "lang": "zh"},
            {"text": "挺好的", "weight": 4, "lang": "zh"},
            {"text": "不错", "weight": 4, "lang": "zh"},
            {"text": "糟糕", "weight": 4, "lang": "zh"},
            {"text": "很糟", "weight": 4, "lang": "zh"},
            {"text": "太难受了", "weight": 4, "lang": "zh"},
            {"text": "受不了", "weight": 4, "lang": "zh"},
            {"text": "难受", "weight": 4, "lang": "zh"},
            {"text": "舒服", "weight": 4, "lang": "zh"},
            {"text": "不舒服", "weight": 4, "lang": "zh"},
            
            # 其他
            {"text": "医生", "weight": 4, "lang": "zh"},
            {"text": "护士", "weight": 4, "lang": "zh"},
            {"text": "医院", "weight": 4, "lang": "zh"},
            {"text": "诊所", "weight": 4, "lang": "zh"},
            {"text": "药店", "weight": 4, "lang": "zh"},
            {"text": "药房", "weight": 4, "lang": "zh"},    
            {"text": "通常", "weight": 4, "lang": "zh"},
            {"text": "问诊", "weight": 4, "lang": "zh"},
            {"text": "看病", "weight": 4, "lang": "zh"},
            {"text": "检查", "weight": 4, "lang": "zh"},
            {"text": "诊断", "weight": 4, "lang": "zh"},
        ]
        
        self._hotwords_cache = medical_terms
        return medical_terms
    
    def cleanup_vocabulary(self):
        """Clean up the created vocabulary to avoid resource leakage"""
        if self.vocabulary_service and self.vocabulary_id:
            try:
                # Only delete vocabulary if it was newly created, not reused
                if not self._vocabulary_is_reused:
                    self.vocabulary_service.delete_vocabulary(self.vocabulary_id)
                    print(f"✓ Deleted medical vocabulary: {self.vocabulary_id}")
                else:
                    print(f"✓ Kept reused medical vocabulary: {self.vocabulary_id}")
                self.vocabulary_id = None
                self._vocabulary_is_reused = False
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
            "核心症状词与确认否定词 (权重5)": len([w for w in hotwords if w['weight'] == 5]),
            "专业术语与口语表达 (权重4)": len([w for w in hotwords if w['weight'] == 4]),
            "时间与程度描述 (权重3)": len([w for w in hotwords if w['weight'] == 3])
        }
        
        return {
            "total_words": len(hotwords),
            "weight_distribution": weight_counts,
            "categories": categories,
            "vocabulary_id": self.vocabulary_id,
            "target_model": self.target_model,
            "is_reused": self._vocabulary_is_reused
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


 