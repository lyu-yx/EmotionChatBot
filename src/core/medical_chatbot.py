"""
Medical Diagnosis Chatbot
------------------------
Specialized chatbot for conducting structured medical consultations with a focus
on Chinese medicine diagnosis. Integrates ASR, LLM, and TTS for a complete medical
consultation experience with logical flow control.
"""

from typing import Optional, List, Dict, Any, Tuple, Set
import os
import time
import threading
import queue
import json
from datetime import datetime
import logging

# Import our component interfaces directly from their modules
from src.asr.speech_recognition_engine import SpeechRecognizer, DashscopeSpeechRecognizer
from src.tts.speech_synthesis import TextToSpeech, StreamingTTSSynthesizer
from src.llm.language_model import LanguageModel, StreamingLanguageModel
from src.emotion.emotion_detector import EmotionDetector, DashscopeEmotionDetector, TextBasedEmotionDetector
from src.emotion.identify import EmotionDetectorCamera
from src.core.SharedQueue import SharedQueue as q
from src.core.SharedLock import SharedLock as lock

logging.basicConfig(level=logging.INFO)

class MedicalDiagnosisChatbot:
    """A specialized medical diagnosis chatbot for structured consultations with logical flow control"""
    
    def __init__(self, 
                 recognizer: Optional[SpeechRecognizer] = None,
                 tts: Optional[TextToSpeech] = None,
                 llm: Optional[LanguageModel] = None,
                 system_prompt: Optional[str] = None,
                 language: str = "zh-cn",
                 use_emotion: bool = False):
        """Initialize the medical diagnosis chatbot
        
        Args:
            recognizer: Speech recognition engine (default: DashscopeSpeechRecognizer)
            tts: Text-to-speech engine (default: StreamingTTSSynthesizer)
            llm: Language model (default: StreamingLanguageModel)
            system_prompt: Optional system prompt to guide the LLM
            language: Language code (default: "zh-cn")
            use_emotion: Whether to consider emotion in responses
        """
        # Default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "你是一个专业的中医问诊助手，需要通过语音交互引导患者完成一次完整的问诊。"
                "你会按照结构化的问诊流程，提问多个方面的问题，收集患者的症状信息。"
                "需要根据患者的回答判断是否需要追问，以及是否跳过某些不相关的问题。"
                "你的目标是用尽量少的轮次获取尽可能全面的病情信息，最终整理成结构化的问诊摘要表格。"
                "请使用专业但通俗易懂的语言，让患者感到温暖和被理解。"
            )
        
        # Initialize speech recognition engine
        self.recognizer = recognizer if recognizer else DashscopeSpeechRecognizer(language=language)
        
        # Initialize streaming text-to-speech engine
        # Choose voice based on language
        voice = "loongstella" if language.startswith("zh") else "xiaomo"
        self.tts = tts if tts else StreamingTTSSynthesizer(voice=voice, model="cosyvoice-v1")
        
        # Initialize streaming language model
        self.llm = llm if llm else StreamingLanguageModel(
            model_name="qwen-turbo", 
            temperature=0.7,
            system_prompt=system_prompt
        )
        
        # Keep track of the current language
        self.language = language
        
        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []
        
        # Maximum number of conversation turns to keep in history
        self.max_history_length = 20
        
        # Patient information collected during consultation
        self.collected_data = {
            # Patient identification
            "gender": None,  # "male", "female"
            "年龄": "",
            
            # T1: Basic medical history
            "慢性病类型": "",
            "治疗方式": "",
            "药物过敏": "",
            
            # T2: Fever and cold
            "是否发热": False,
            "发热温度": "",
            "发热时间段": "",
            "是否怕冷": False,
            "是否怯寒": "",
            "是否出汗": False,
            "汗液类型": "",
            
            # T3: Head symptoms
            "是否头痛": False,
            "头痛部位": "",
            "头痛类型": "",
            "是否头晕": False,
            "头晕类型": "",
            "是否伴随恶心": False,
            
            # T4: Five senses (eyes, ears, nose)
            "眼部不适": "",
            "耳部症状": "",
            "鼻部症状": "",
            
            # T5: Throat and respiratory
            "咽部不适": "",
            "是否咳嗽": False,
            "咳嗽类型": "",
            "是否咳痰": False,
            "痰颜色": "",
            "痰质地": "",
            "是否易咳出": "",
            "是否胸闷": False,
            "是否心悸": False,
            
            # T6: Appetite and hydration
            "食欲情况": "",
            "口腔症状": "",
            "饮水习惯": "",
            
            # T7: Excretory system
            "小便情况": "",
            "尿色": "",
            "大便频率": "",
            "大便性状": "",
            "腹痛位置": "",
            "腹痛缓解": "",
            
            # T8: Sleep, mood, skin
            "睡眠质量": "",
            "梦境情况": "",
            "情绪状态": "",
            "皮肤症状": "",
            
            # T9: Female specific (only for female patients)
            "月经周期": "",
            "颜色与量": "",
            "是否痛经": False,
            "白带情况": ""
        }
        
        # Store raw responses for each topic
        self.raw_responses = {
            "T1-基础信息": "",
            "T2-发热寒热": "",
            "T3-头痛头晕": "",
            "T4-五官": "",
            "T5-咽喉与咳嗽": "",
            "T6-食欲饮水": "",
            "T7-大小便与腹痛": "",
            "T8-睡眠": "",
            "T8.1-情绪": "",
            "T8.2-皮肤": "",
            "T9-女性月经": ""
        }
        
        # Define the consultation flow with logic
        self.consultation_flow = [
            {
                "id": "T1-基础信息",
                "question": "首先，请告诉我您的性别和年龄？",
                "fields": ["gender", "年龄"],
                "condition": "始终提问",
                "follow_ups": [
                    {
                        "condition": "始终提问",
                        "question": "您有没有慢性病，比如高血压、高血糖、高血脂、胃病等？是否在治疗？是否有药物过敏？",
                        "fields": ["慢性病类型", "治疗方式", "药物过敏"]
                    }
                ]
            },
            {
                "id": "T2-发热寒热",
                "question": "最近有没有发烧或怕冷？",
                "fields": ["是否发热", "是否怕冷"],
                "condition": "始终提问",
                "follow_ups": [
                    {
                        "condition": "是否发热 == True",
                        "question": "发烧大概多少度？哪个时间段最明显？",
                        "fields": ["发热温度", "发热时间段"]
                    },
                    {
                        "condition": "是否怕冷 == True",
                        "question": "穿衣服后能缓解怕冷吗？",
                        "fields": ["是否怯寒"]
                    },
                    {
                        "condition": "是否发热 == True or 是否怕冷 == True",
                        "question": "有没有出汗？是清水汗还是黏汗？",
                        "fields": ["是否出汗", "汗液类型"]
                    }
                ]
            },
            {
                "id": "T3-头痛头晕",
                "question": "最近有没有头痛或头晕？",
                "fields": ["是否头痛", "是否头晕"],
                "condition": "始终提问",
                "follow_ups": [
                    {
                        "condition": "是否头痛 == True",
                        "question": "头痛在什么部位？是胀痛、刺痛还是抽痛？",
                        "fields": ["头痛部位", "头痛类型"]
                    },
                    {
                        "condition": "是否头晕 == True",
                        "question": "是头部昏沉还是天旋地转？有没有伴随恶心呕吐？",
                        "fields": ["头晕类型", "是否伴随恶心"]
                    }
                ]
            },
            {
                "id": "T4-五官",
                "question": "眼睛有没有不适，比如干涩、发痒、流泪或视力问题？",
                "fields": ["眼部不适"],
                "condition": "始终提问",
                "follow_ups": [
                    {
                        "condition": "始终提问",
                        "question": "请问是否有耳鸣或听力问题？",
                        "fields": ["耳部症状"]
                    },
                    {
                        "condition": "始终提问",
                        "question": "鼻子是否有鼻塞、流涕？",
                        "fields": ["鼻部症状"]
                    }
                ]
            },
            {
                "id": "T5-咽喉与咳嗽",
                "question": "喉咙是否干、痒、疼或堵？最近有没有咳嗽？",
                "fields": ["咽部不适", "是否咳嗽"],
                "condition": "始终提问",
                "follow_ups": [
                    {
                        "condition": "是否咳嗽 == True",
                        "question": "咳嗽是间断还是持续？有痰吗？",
                        "fields": ["咳嗽类型", "是否咳痰"]
                    },
                    {
                        "condition": "是否咳痰 == True",
                        "question": "痰的颜色和质地如何？容易咳出吗？",
                        "fields": ["痰颜色", "痰质地", "是否易咳出"]
                    },
                    {
                        "condition": "是否咳嗽 == True",
                        "question": "是否伴随胸闷或心悸？",
                        "fields": ["是否胸闷", "是否心悸"]
                    }
                ]
            },
            {
                "id": "T6-食欲饮水",
                "question": "最近食欲如何？有没有偏好吃冷食或热食？",
                "fields": ["食欲情况"],
                "condition": "始终提问",
                "follow_ups": [
                    {
                        "condition": "始终提问",
                        "question": "有没有口苦、口干、反酸等口腔症状？",
                        "fields": ["口腔症状"]
                    },
                    {
                        "condition": "始终提问",
                        "question": "平时喝水习惯是怎样的？喜欢热水还是冷水？",
                        "fields": ["饮水习惯"]
                    }
                ]
            },
            {
                "id": "T7-大小便与腹痛",
                "question": "小便通畅吗？颜色如何？",
                "fields": ["小便情况", "尿色"],
                "condition": "始终提问",
                "follow_ups": [
                    {
                        "condition": "始终提问",
                        "question": "大便情况如何？次数和形状是否正常？",
                        "fields": ["大便频率", "大便性状"]
                    },
                    {
                        "condition": "始终提问",
                        "question": "有没有腹痛或腹胀？排便后是否有缓解？",
                        "fields": ["腹痛位置", "腹痛缓解"]
                    }
                ]
            },
            {
                "id": "T8-睡眠",
                "question": "最近睡眠是否良好？容易入睡吗？",
                "fields": ["睡眠质量"],
                "condition": "始终提问",
                "follow_ups": [
                    {
                        "condition": "始终提问",
                        "question": "有没有多梦或早醒的情况？",
                        "fields": ["梦境情况"]
                    }
                ]
            },
            {
                "id": "T8.1-情绪",
                "question": "最近情绪状态如何？有没有烦躁、发怒或闷闷不乐的情况？",
                "fields": ["情绪状态"],
                "condition": "始终提问",
                "follow_ups": []
            },
            {
                "id": "T8.2-皮肤",
                "question": "皮肤有没有异常，比如瘙痒、红疹、湿疹等？",
                "fields": ["皮肤症状"],
                "condition": "始终提问",
                "follow_ups": []
            },
            {
                "id": "T9-女性月经",
                "question": "请问您的月经是否规律？颜色、量是否正常？",
                "fields": ["月经周期", "颜色与量"],
                "condition": "gender == 'female'",
                "follow_ups": [
                    {
                        "condition": "始终提问",
                        "question": "有没有痛经？白带情况如何？",
                        "fields": ["是否痛经", "白带情况"]
                    }
                ]
            }
        ]
        
        # Current topic index
        self.current_topic_index = 0
        
        # Current follow-up index within the current topic
        self.current_followup_index = -1
        
        # Set of topics that have been completed
        self.completed_topics: Set[str] = set()
        
        # Flag indicating if we need to repeat the current question
        self.repeat_current_question = False
        
        # Flag to track if the user's response was related to the question
        self.response_relevant = True
        
        # Transcript of the entire conversation
        self.conversation_transcript = []
        
        # Flag to indicate if TTS is currently active
        self.is_speaking = False
        
        # Flag to judge whether the bot is active
        self.is_active = False
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Store the message
        self.queue = q()
        
        # Control the listen_continuous thread
        self.listen_thread = threading.Thread(target=self.listen_continuous)
        self.listen_thread.daemon = True
        self.listen_interrupt_stop = threading.Event()
        
        # Flag to indicate if consultation is complete
        self.consultation_complete = False
        
        # Whether to use emotion in responses
        self.use_emotion = use_emotion
        
        # Lock to avoid listen confliction
        self.listen_lock = lock()
        
        # Try to warm up the models with simple queries
        try:
            # Warm up LLM with a simple request
            self.llm.generate_response("你好", [])
            print("LLM model warmed up")
        except Exception as e:
            print(f"LLM warmup failed (not critical): {e}")
            
        print("Medical Diagnosis Chatbot initialized with logical flow control")
    
    def listen_continuous(self):
        """Continuously listen for user input in a background thread"""
        while True:
            with self.listen_lock:
                if not self.listen_interrupt_stop.is_set():
                    try:
                        # Check if speech is ongoing and should be interrupted
                        if self.is_speaking:
                            # Let the speech complete for medical consultation accuracy
                            time.sleep(0.1)
                            continue
                        
                        # Recognize speech
                        result = self.recognizer.recognize_from_microphone()
                        if result and result["text"] != '':
                            self.queue.put(result)
                    except Exception as e:
                        print(f"Listen thread exception: {e}")
            time.sleep(0.05)  # Small sleep to prevent CPU overuse
    def listen(self) -> Dict[str, Any]:
        """Listen for user input via microphone

        Returns:
            Dict with recognition results
        """
        # Don't listen if we're currently speaking
        with self.lock:
            if self.is_speaking:
                print("Speaking in progress, postponing listening...")
                time.sleep(0.5)  # Small delay to check again

        return self.recognizer.recognize_from_microphone()

    def speak(self, text: str) -> Dict[str, Any]:
        """Convert text to speech and speak it, with protection against recording
        
        Args:
            text: The text to speak
            
        Returns:
            Dict with speech result
        """
        try:
            # Set the speaking flag to prevent listening while speaking
            with self.lock:
                self.is_speaking = True
                
            # Speak the text
            result = self.tts.speak(text)
            print(f"Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
                
            return result
        finally:
            # Make sure to reset the flag even if an error occurs
            with self.lock:
                self.is_speaking = False
    
    def evaluate_condition(self, condition: str) -> bool:
        """Evaluate a condition string against the collected data
        
        Args:
            condition: A condition string to evaluate
            
        Returns:
            Boolean indicating if the condition is met
        """
        if condition == "始终提问":
            return True
            
        if condition.startswith("gender =="):
            gender_value = condition.split("==")[1].strip().strip("'").strip('"')
            return self.collected_data["gender"] == gender_value
            
        # For other conditions, evaluate as Python expressions
        try:
            # Replace field names with their values
            for field, value in self.collected_data.items():
                # Properly format string values for evaluation
                if isinstance(value, str):
                    condition = condition.replace(field, f"'{value}'")
                else:
                    condition = condition.replace(field, str(value))
                    
            # Evaluate the condition
            return eval(condition)
        except Exception as e:
            print(f"Error evaluating condition '{condition}': {e}")
            # Default to True if there's an error
            return True
    
    def get_current_topic(self) -> Dict[str, Any]:
        """Get the current consultation topic
        
        Returns:
            The current topic dictionary
        """
        if self.current_topic_index < len(self.consultation_flow):
            return self.consultation_flow[self.current_topic_index]
        return None
    
    def get_current_question(self) -> str:
        """Get the current question to ask
        
        Returns:
            The current question text
        """
        # If we need to repeat due to irrelevant response
        if self.repeat_current_question and not self.response_relevant:
            return "请您直接回答当前问题，" + self.get_base_question()
            
        # Otherwise get the normal question
        return self.get_base_question()
    
    def get_base_question(self) -> str:
        """Get the base question without any prompt to focus
        
        Returns:
            The base question text
        """
        current_topic = self.get_current_topic()
        
        # If we're at a follow-up question
        if self.current_followup_index >= 0 and current_topic and "follow_ups" in current_topic:
            if self.current_followup_index < len(current_topic["follow_ups"]):
                follow_up = current_topic["follow_ups"][self.current_followup_index]
                if self.evaluate_condition(follow_up["condition"]):
                    return follow_up["question"]
                else:
                    # Skip this follow-up since condition is not met
                    self.current_followup_index += 1
                    return self.get_base_question()
            else:
                # No more follow-ups, move to next topic
                self.current_topic_index += 1
                self.current_followup_index = -1
                self.completed_topics.add(current_topic["id"])
                return self.get_base_question()
        
        # If we're at a main topic question
        if current_topic:
            # Check if this topic's condition is met
            if self.evaluate_condition(current_topic["condition"]):
                # Get conversational introduction for this topic
                intro_text = self.get_topic_conversational_intro(current_topic["id"])
                return f"{intro_text}{current_topic['question']}"
            else:
                # Skip to next topic
                self.current_topic_index += 1
                return self.get_base_question()
        
        # If we've reached the end of all topics
        return self.get_closing_question()
    
    def get_topic_conversational_intro(self, topic_id: str) -> str:
        """Generate a conversational introduction for a topic
        
        Args:
            topic_id: The topic ID
            
        Returns:
            Conversational introduction text
        """
        # Map topic IDs to conversational introductions
        topic_intros = {
            "T1-基础信息": "让我们先了解一下您的基本情况，",
            "T2-发热寒热": "关于体温情况，",
            "T3-头痛头晕": "接下来谈谈头部症状，",
            "T4-五官": "再来了解一下您的五官情况，",
            "T5-咽喉与咳嗽": "关于呼吸系统，",
            "T6-食欲饮水": "现在聊聊您的饮食情况，",
            "T7-大小便与腹痛": "关于消化排泄系统，",
            "T8-睡眠": "让我们谈谈睡眠质量，",
            "T8.1-情绪": "关于近期的情绪状态，",
            "T8.2-皮肤": "皮肤健康方面，",
            "T9-女性月经": "对于女性患者，我还需要了解一下，",
            "T10-性别确认": ""  # No intro needed as this is moved to T1
        }
        
        return topic_intros.get(topic_id, "")
    
    def get_closing_question(self) -> str:
        """Get closing question or statement after all topics
        
        Returns:
            Closing statement or question
        """
        return "好的，我们已经完成了主要问诊。感谢您的配合，现在我已经收集完所有必要的信息。您还有什么补充的症状或问题吗？"
    
    def extract_data_from_response(self, response: str) -> Dict[str, Any]:
        """Extract structured data from the user's response
        
        Args:
            response: The user's response text
            
        Returns:
            Dictionary of extracted data
        """
        current_topic = self.get_current_topic()
        
        # If we've completed all topics
        if not current_topic:
            return {}
            
        # For gender and age confirmation question, extract directly
        if current_topic["id"] == "T1-基础信息" and self.current_followup_index == -1:
            extracted_data = {}
            
            # Extract gender information
            if "女" in response:
                extracted_data["gender"] = "female"
            elif "男" in response:
                extracted_data["gender"] = "male"
            
            # Try to extract age information using common patterns
            import re
            age_patterns = [
                r'(\d+)\s*岁',  # For patterns like "35岁", "35 岁"
                r'年龄\s*(\d+)',  # For patterns like "年龄35", "年龄 35" 
                r'(\d+)\s*年',  # For patterns like "35年"
                r'(\d+)\s*[年|周]?\s*[岁|歲]',  # Various combinations
                r'age\s*[is|am]?\s*(\d+)',  # English patterns
            ]
            
            for pattern in age_patterns:
                age_match = re.search(pattern, response)
                if age_match:
                    age_value = age_match.group(1)
                    # Validate age is reasonable (1-120)
                    try:
                        age_int = int(age_value)
                        if 1 <= age_int <= 120:
                            extracted_data["年龄"] = age_value
                            break
                    except ValueError:
                        continue
            
            if extracted_data:
                return extracted_data
            
        # For gender confirmation question, extract gender directly
        if current_topic["id"] == "T10-性别确认":
            if "女" in response:
                return {"gender": "female"}
            elif "男" in response:
                return {"gender": "male"}
            else:
                return {"gender": "unknown"}
        
        # Check for negations before checking for keywords
        has_negation = any(neg in response for neg in ["没有", "不", "无", "不存在", "否认"])
        
        # For simple yes/no answers about symptoms, extract directly
        if current_topic["id"] == "T2-发热寒热" and self.current_followup_index == -1:
            # Process fever keywords, but check for negations first
            if any(keyword in response for keyword in ["发烧", "发热"]) and not has_negation:
                return {"是否发热": True}
            elif any(keyword in response for keyword in ["发烧", "发热"]) and has_negation:
                return {"是否发热": False}
                
            # Process cold sensitivity keywords
            if any(keyword in response for keyword in ["怕冷", "畏寒"]) and not has_negation:
                return {"是否怕冷": True}
            elif any(keyword in response for keyword in ["怕冷", "畏寒"]) and has_negation:
                return {"是否怕冷": False}
            
        # For other topics with boolean questions
        for topic_with_booleans in ["T3-头痛头晕", "T5-咽喉与咳嗽"]:
            if current_topic["id"] == topic_with_booleans and self.current_followup_index == -1:
                extracted = self.extract_boolean_answers(response, current_topic["id"], has_negation)
                if extracted:
                    return extracted
            
        # Determine which fields we need to extract
        if self.current_followup_index >= 0 and "follow_ups" in current_topic:
            if self.current_followup_index < len(current_topic["follow_ups"]):
                fields = current_topic["follow_ups"][self.current_followup_index]["fields"]
            else:
                fields = []
        else:
            fields = current_topic["fields"]
            
        # Create a system prompt for the LLM to extract data
        system_prompt = (
            "你是一个医疗问诊数据提取专家。你的任务是从患者回答中提取特定字段的信息。"
            "对于每个字段，如果患者提供了相关信息，请提取出来。"
            "对于布尔类型字段(是否XX)，请判断为True或False。"
            "对于其他字段，请提供提取到的值或空字符串。"
            "如果患者回答中包含否定词（如'没有'、'不'、'无'），请正确判断为False。"
            "请以JSON格式返回结果，不要有任何多余文字。例如：\n"
            "{\n"
            "  \"是否头痛\": false,\n"
            "  \"头痛部位\": \"\",\n"
            "  \"是否发热\": true\n"
            "}"
        )
        
        # Create the extraction prompt
        extraction_prompt = f"患者的回答：\"{response}\"\n\n请从这个回答中提取以下字段的信息：{fields}"
        
        result = self.llm.generate_response(
            user_input=extraction_prompt,
            conversation_history=[{"role": "system", "content": system_prompt}]
        )
        
        extracted_data = {}
        if result["success"]:
            try:
                # Try to parse the JSON response
                # First find the JSON part in the response (it might have explanations)
                response_text = result["response"]
                
                # Look for JSON within the response
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    extracted_data = json.loads(json_str)
                    
                    # Convert string "True"/"False" to boolean values
                    for key, value in extracted_data.items():
                        if isinstance(value, str):
                            if value.lower() == "true":
                                extracted_data[key] = True
                            elif value.lower() == "false":
                                extracted_data[key] = False
                            
                    print(f"Extracted data: {extracted_data}")
                else:
                    # Fallback for non-JSON responses
                    print("Could not find valid JSON in the response.")
                    
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from response: {result['response']}")
            except Exception as e:
                print(f"Error processing extraction result: {e}")
        
        return extracted_data
    
    def extract_boolean_answers(self, response: str, topic_id: str, has_negation: bool = False) -> Dict[str, Any]:
        """Extract boolean answers for specific topics
        
        Args:
            response: The user's response text
            topic_id: The ID of the current topic
            has_negation: Whether the response contains negation words
            
        Returns:
            Dictionary of boolean field values
        """
        extracted = {}
        
        # Handle each topic specifically
        if topic_id == "T3-头痛头晕":
            headache_keywords = ["头痛", "头疼"]
            dizziness_keywords = ["头晕", "天旋地转", "晕眩"]
            
            # Check for headache symptoms
            if any(keyword in response for keyword in headache_keywords):
                extracted["是否头痛"] = not has_negation
                
            # Check for dizziness symptoms  
            if any(keyword in response for keyword in dizziness_keywords):
                extracted["是否头晕"] = not has_negation
                
        elif topic_id == "T5-咽喉与咳嗽":
            cough_keywords = ["咳嗽", "咳", "咳痰"]
            throat_keywords = ["咽干", "喉咙痛", "喉咙干", "喉痛", "咽痛"]
            
            # Check for cough symptoms
            if any(keyword in response for keyword in cough_keywords):
                extracted["是否咳嗽"] = not has_negation
                
            # Check for throat symptoms
            if any(keyword in response for keyword in throat_keywords):
                if not has_negation:
                    extracted["咽部不适"] = "有不适"
                else:
                    extracted["咽部不适"] = ""
                
        return extracted
    
    def extract_gender_from_response(self, response: str) -> str:
        """Extract gender information from response text
        
        Args:
            response: The user's response text
            
        Returns:
            String representing gender: 'male', 'female', or 'unknown'
        """
        # Look for gender indicators in the response
        female_indicators = ["女", "女性", "女士", "姨妈", "月经", "怀孕", "妇科"]
        male_indicators = ["男", "男性", "先生", "前列腺"]
        
        # Check for explicit indicators
        for indicator in female_indicators:
            if indicator in response:
                return "female"
                
        for indicator in male_indicators:
            if indicator in response:
                return "male"
        
        # If no clear indicators, use LLM to determine
        system_prompt = "你是一个分析系统，需要从文本中判断说话者的性别。"
        analysis_prompt = f"根据以下文本，判断说话者是男性还是女性（如果无法确定，请回答'未知'）：\n\n{response}"
        
        result = self.llm.generate_response(
            user_input=analysis_prompt,
            conversation_history=[{"role": "system", "content": system_prompt}]
        )
        
        if result["success"]:
            if "女" in result["response"]:
                return "female"
            elif "男" in result["response"]:
                return "male"
        
        # Default if analysis fails
        return "unknown"
    
    def detect_gender(self, response: str) -> None:
        """Detect the patient's gender from their response
        
        Args:
            response: Patient's response text
        """
        gender = self.extract_gender_from_response(response)
        if gender:
            self.collected_data["gender"] = gender
            print(f"Detected gender: {gender}")
    
    def process_response(self, response: str) -> bool:
        """Process the patient's response and advance the consultation
        
        Args:
            response: The user's response text
            
        Returns:
            Boolean indicating if the consultation should continue
        """
        # Skip processing if consultation is complete
        if self.consultation_complete:
            return False
            
        # Check if the response is relevant to the current question
        self.response_relevant = self.is_response_relevant(response)
        
        if not self.response_relevant:
            # If the response is not relevant, we'll repeat the question
            self.repeat_current_question = True
            return True
            
        # Reset the repeat flag
        self.repeat_current_question = False
        
        # Store the response in raw_responses for the current topic
        current_topic = self.get_current_topic()
        if current_topic and current_topic["id"] in self.raw_responses:
            if self.current_followup_index >= 0 and "follow_ups" in current_topic:
                # This is a follow-up response, append to the main topic
                if self.raw_responses[current_topic["id"]]:
                    self.raw_responses[current_topic["id"]] += " | " + response
                else:
                    self.raw_responses[current_topic["id"]] = response
            else:
                # This is a main topic response
                self.raw_responses[current_topic["id"]] = response
        
        # Special handling for gender detection - we do this regardless of the current topic
        # to ensure we capture gender info whenever mentioned
        if self.collected_data["gender"] is None or self.collected_data["gender"] == "unknown":
            self.detect_gender(response)
            
        # Extract data from the response
        extracted_data = self.extract_data_from_response(response)
        
        # Update the collected data
        self.collected_data.update(extracted_data)
        
        # Move to the next question
        if current_topic:
            # If we're at a main topic question
            if self.current_followup_index == -1:
                # Check if there are follow-ups for this topic
                if "follow_ups" in current_topic and current_topic["follow_ups"]:
                    # Move to the first follow-up that has its condition met
                    self.current_followup_index = 0
                    
                    # Skip follow-ups whose conditions are not met
                    while (self.current_followup_index < len(current_topic["follow_ups"]) and 
                           not self.evaluate_condition(current_topic["follow_ups"][self.current_followup_index]["condition"])):
                        self.current_followup_index += 1
                        
                    # If we've gone through all follow-ups, move to the next topic
                    if self.current_followup_index >= len(current_topic["follow_ups"]):
                        self.current_topic_index += 1
                        self.current_followup_index = -1
                        self.completed_topics.add(current_topic["id"])
                else:
                    # No follow-ups, move directly to the next topic
                    self.current_topic_index += 1
                    self.completed_topics.add(current_topic["id"])
            else:
                # We're in a follow-up, move to the next one
                self.current_followup_index += 1
                
                # Skip follow-ups whose conditions are not met
                while (self.current_followup_index < len(current_topic["follow_ups"]) and 
                       not self.evaluate_condition(current_topic["follow_ups"][self.current_followup_index]["condition"])):
                    self.current_followup_index += 1
                    
                # If we've gone through all follow-ups, move to the next topic
                if self.current_followup_index >= len(current_topic["follow_ups"]):
                    self.current_topic_index += 1
                    self.current_followup_index = -1
                    self.completed_topics.add(current_topic["id"])
                
        # Check if we've completed all topics
        if self.current_topic_index >= len(self.consultation_flow):
            return False
            
        return True
    
    def generate_consultation_summary(self) -> str:
        """Generate a structured summary of the consultation
        
        Returns:
            Formatted summary table
        """
        # Create header for the table
        summary = "## 问诊摘要表\n\n"
        summary += "| 类别 | 内容摘要 |\n"
        summary += "|------|--------|\n"
        
        # Add patient basic information
        gender_display = {
            "male": "男性",
            "female": "女性",
            "unknown": "未确定"
        }
        
        basic_info = f"性别：{gender_display.get(self.collected_data['gender'], '未确定')}"
        if self.collected_data.get("年龄", ""):
            basic_info += f"，年龄：{self.collected_data['年龄']}"
        
        summary += f"| 基本信息 | {basic_info} |\n"
        
        # Create more specific summaries for each category
        topic_summaries = self.create_detailed_summaries()
        
        # Add rows to the table
        for category, content in topic_summaries.items():
            # Skip female-specific for male patients
            if category == "女性生理情况" and self.collected_data["gender"] == "male":
                summary += f"| {category} | 不适用（男性患者） |\n"
            else:
                content = content or "未提供信息"
                summary += f"| {category} | {content} |\n"
        
        # Add consultation timestamp
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary += f"\n**问诊时间**：{now}\n"
        
        # Add raw collected data in JSON format
        summary += "\n## 原始数据\n\n```json\n"
        summary += json.dumps(self.collected_data, ensure_ascii=False, indent=2)
        summary += "\n```\n"
        
        return summary
    
    def create_detailed_summaries(self) -> Dict[str, str]:
        """Create detailed summaries for each topic
        
        Returns:
            Dictionary of category summaries
        """
        summaries = {}
        
        # Chronic conditions summary
        chronic_info = []
        if self.collected_data.get("慢性病类型", ""):
            chronic_info.append(f"慢性病：{self.collected_data['慢性病类型']}")
        if self.collected_data.get("治疗方式", ""):
            chronic_info.append(f"治疗：{self.collected_data['治疗方式']}")
        if self.collected_data.get("药物过敏", ""):
            chronic_info.append(f"药物过敏：{self.collected_data['药物过敏']}")
        
        chronic_summary = "；".join(chronic_info) if chronic_info else "无慢性病及药物过敏"
        summaries["基础病史"] = chronic_summary
        
        # Fever and cold summary
        fever_info = []
        if self.collected_data.get("是否发热", False):
            temp_details = []
            if self.collected_data.get("发热温度", ""):
                temp_details.append(f"{self.collected_data['发热温度']}")
            if self.collected_data.get("发热时间段", ""):
                temp_details.append(f"{self.collected_data['发热时间段']}明显")
            
            fever_desc = "有发热"
            if temp_details:
                fever_desc += f"（{', '.join(temp_details)}）"
            fever_info.append(fever_desc)
        else:
            fever_info.append("无发热")
            
        if self.collected_data.get("是否怕冷", False):
            cold_desc = "有怕冷"
            if self.collected_data.get("是否怯寒", ""):
                cold_desc += f"（{self.collected_data['是否怯寒']}）"
            fever_info.append(cold_desc)
        else:
            fever_info.append("无怕冷")
            
        if self.collected_data.get("是否出汗", False) and self.collected_data.get("汗液类型", ""):
            fever_info.append(f"出汗（{self.collected_data['汗液类型']}）")
        
        summaries["寒热发热情况"] = "；".join(fever_info)
        
        # Head symptoms summary
        head_info = []
        if self.collected_data.get("是否头痛", False):
            pain_details = []
            if self.collected_data.get("头痛部位", ""):
                pain_details.append(self.collected_data["头痛部位"])
            if self.collected_data.get("头痛类型", ""):
                pain_details.append(self.collected_data["头痛类型"])
                
            head_desc = "有头痛"
            if pain_details:
                head_desc += f"（{', '.join(pain_details)}）"
            head_info.append(head_desc)
        else:
            head_info.append("无头痛")
            
        if self.collected_data.get("是否头晕", False):
            dizzy_details = []
            if self.collected_data.get("头晕类型", ""):
                dizzy_details.append(self.collected_data["头晕类型"])
            if self.collected_data.get("是否伴随恶心", True):
                dizzy_details.append("伴有恶心")
                
            dizzy_desc = "有头晕"
            if dizzy_details:
                dizzy_desc += f"（{', '.join(dizzy_details)}）"
            head_info.append(dizzy_desc)
        else:
            head_info.append("无头晕")
        
        # Five senses summary
        sense_info = []
        if self.collected_data.get("眼部不适", ""):
            sense_info.append(f"眼部：{self.collected_data['眼部不适']}")
        else:
            sense_info.append("眼部正常")
            
        if self.collected_data.get("耳部症状", ""):
            sense_info.append(f"耳部：{self.collected_data['耳部症状']}")
        else:
            sense_info.append("耳部正常")
            
        if self.collected_data.get("鼻部症状", ""):
            sense_info.append(f"鼻部：{self.collected_data['鼻部症状']}")
        else:
            sense_info.append("鼻部正常")
        
        head_sense_summary = "；".join(head_info) + "。" + "；".join(sense_info)
        summaries["头面五官"] = head_sense_summary
        
        # Respiratory symptoms summary
        resp_info = []
        if self.collected_data.get("咽部不适", ""):
            resp_info.append(f"咽部：{self.collected_data['咽部不适']}")
        else:
            resp_info.append("咽部正常")
            
        if self.collected_data.get("是否咳嗽", False):
            cough_details = []
            if self.collected_data.get("咳嗽类型", ""):
                cough_details.append(self.collected_data["咳嗽类型"])
                
            if self.collected_data.get("是否咳痰", False):
                phlegm_desc = "有痰"
                phlegm_details = []
                if self.collected_data.get("痰颜色", ""):
                    phlegm_details.append(self.collected_data["痰颜色"])
                if self.collected_data.get("痰质地", ""):
                    phlegm_details.append(self.collected_data["痰质地"])
                if self.collected_data.get("是否易咳出", ""):
                    phlegm_details.append(f"{'易' if '易' in self.collected_data['是否易咳出'] else '难'}咳出")
                
                if phlegm_details:
                    phlegm_desc += f"（{', '.join(phlegm_details)}）"
                cough_details.append(phlegm_desc)
                
            if self.collected_data.get("是否胸闷", True):
                cough_details.append("伴有胸闷")
            if self.collected_data.get("是否心悸", True):
                cough_details.append("伴有心悸")
                
            cough_desc = "有咳嗽"
            if cough_details:
                cough_desc += f"（{', '.join(cough_details)}）"
            resp_info.append(cough_desc)
        else:
            resp_info.append("无咳嗽")
        
        summaries["呼吸咽喉系统"] = "；".join(resp_info)
        
        # Digestive symptoms summary
        digest_info = []
        if self.collected_data.get("食欲情况", ""):
            digest_info.append(f"食欲：{self.collected_data['食欲情况']}")
        if self.collected_data.get("口腔症状", ""):
            digest_info.append(f"口腔：{self.collected_data['口腔症状']}")
        if self.collected_data.get("饮水习惯", ""):
            digest_info.append(f"饮水：{self.collected_data['饮水习惯']}")
            
        summaries["消化系统"] = "；".join(digest_info) if digest_info else "未提供信息"
        
        # Excretory symptoms summary
        excret_info = []
        if self.collected_data.get("小便情况", ""):
            urine_desc = f"小便：{self.collected_data['小便情况']}"
            if self.collected_data.get("尿色", ""):
                urine_desc += f"，{self.collected_data['尿色']}"
            excret_info.append(urine_desc)
            
        if self.collected_data.get("大便频率", "") or self.collected_data.get("大便性状", ""):
            stool_desc = "大便："
            if self.collected_data.get("大便频率", ""):
                stool_desc += f"{self.collected_data['大便频率']}"
            if self.collected_data.get("大便性状", ""):
                stool_desc += f"，{self.collected_data['大便性状']}"
            excret_info.append(stool_desc)
            
        if self.collected_data.get("腹痛位置", ""):
            pain_desc = f"腹痛：{self.collected_data['腹痛位置']}"
            if self.collected_data.get("腹痛缓解", ""):
                pain_desc += f"，{self.collected_data['腹痛缓解']}"
            excret_info.append(pain_desc)
            
        summaries["排泄系统"] = "；".join(excret_info) if excret_info else "未提供信息"
        
        # Sleep, mood, skin summary
        well_info = []
        
        if self.collected_data.get("睡眠质量", ""):
            sleep_desc = f"睡眠：{self.collected_data['睡眠质量']}"
            if self.collected_data.get("梦境情况", ""):
                sleep_desc += f"，{self.collected_data['梦境情况']}"
            well_info.append(sleep_desc)
            
        if self.collected_data.get("情绪状态", ""):
            well_info.append(f"情绪：{self.collected_data['情绪状态']}")
            
        if self.collected_data.get("皮肤症状", ""):
            well_info.append(f"皮肤：{self.collected_data['皮肤症状']}")
        else:
            well_info.append("皮肤无异常")
            
        summaries["睡眠情绪皮肤"] = "；".join(well_info) if well_info else "未提供信息"
        
        # Female specific symptoms summary
        if self.collected_data["gender"] == "female":
            female_info = []
            if self.collected_data.get("月经周期", ""):
                period_desc = f"月经周期：{self.collected_data['月经周期']}"
                if self.collected_data.get("颜色与量", ""):
                    period_desc += f"，{self.collected_data['颜色与量']}"
                female_info.append(period_desc)
                
            if self.collected_data.get("是否痛经", False):
                female_info.append("有痛经")
                
            if self.collected_data.get("白带情况", ""):
                female_info.append(f"白带：{self.collected_data['白带情况']}")
                
            summaries["女性生理情况"] = "；".join(female_info) if female_info else "未提供信息"
        
        return summaries
    
    def run_once(self) -> Dict[str, Any]:
        """Run one complete interaction cycle
        
        Returns:
            Dict with interaction results
        """
        result = {
            "user_input": "",
            "response": "",
            "success": False,
            "error": None,
            "consultation_complete": self.consultation_complete
        }
        
        try:
            # Ensure gender is determined before ending consultation
            if self.current_topic_index >= len(self.consultation_flow) - 1 and not self.consultation_complete:
                if self.collected_data["gender"] is None or self.collected_data["gender"] == "unknown":
                    # Force ask gender question if not determined yet
                    self.current_topic_index = len(self.consultation_flow) - 1  # Point to gender question
                    self.current_followup_index = -1
            
            # Get the next question
            question = self.get_current_question()
            
            # If no more questions, end the consultation
            if self.current_topic_index >= len(self.consultation_flow) and not self.consultation_complete:
                print("All topics complete, generating summary...")
                self.consultation_complete = True
                summary = self.generate_consultation_summary()
                print(summary)
                
                # Speak a completion message
                completion_message = "非常感谢您的配合，问诊已经完成。我已经为您整理了一份问诊摘要，稍后会交给医生进行专业诊断。祝您早日康复！"
                self.speak(completion_message)
                
                result["response"] = completion_message
                result["consultation_complete"] = True
                result["success"] = True
                return result
            
            # If consultation is already complete, just handle additional questions
            if self.consultation_complete:
                # Get any final questions from the patient
                print("Waiting for any final questions from patient...")
                listen_result = self.queue.get()
                
                if not listen_result["success"]:
                    result["error"] = listen_result["error"]
                    return result
                
                user_input = listen_result["text"]
                result["user_input"] = user_input
                
                # Check if user wants to end
                if any(exit_word in user_input for exit_word in ["结束", "退出", "谢谢", "再见"]):
                    result["response"] = "感谢您的配合，问诊已结束。祝您健康！"
                    self.speak(result["response"])
                    result["success"] = True
                    return result
                
                # Process final questions with LLM
                llm_result = self.llm.generate_response(
                    user_input=user_input,
                    conversation_history=self.conversation_history
                )
                
                if llm_result["success"]:
                    result["response"] = llm_result["response"]
                    self.speak(result["response"])
                    
                    # Update conversation history
                    self.conversation_history.append({"role": "user", "content": user_input})
                    self.conversation_history.append({"role": "assistant", "content": llm_result["response"]})
                    
                    # Trim history if needed
                    if len(self.conversation_history) > self.max_history_length * 2:
                        self.conversation_history = self.conversation_history[-self.max_history_length*2:]
                    
                    result["success"] = True
                else:
                    result["error"] = llm_result["error"]
                
                return result
            
            # Speak the current question
            current_topic = self.get_current_topic()
            topic_name = current_topic["id"] if current_topic else "Closing"
            print(f"Current topic: {topic_name}, Follow-up index: {self.current_followup_index}")
            self.speak(question)
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": question})
            
            # Listen for response
            print("Waiting for patient response...")
            listen_result = self.queue.get()
            
            if not listen_result["success"]:
                result["error"] = listen_result["error"]
                return result
                
            user_input = listen_result["text"]
            result["user_input"] = user_input
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Trim history if needed
            if len(self.conversation_history) > self.max_history_length * 2:
                self.conversation_history = self.conversation_history[-self.max_history_length*2:]
            
            # Process the response
            should_continue = self.process_response(user_input)
            
            if not should_continue:
                # Mark consultation as complete
                self.consultation_complete = True
                summary = self.generate_consultation_summary()
                print("="*50)
                print("医疗问诊摘要:")
                print(summary)
                print("="*50)
                
                # Speak a completion message
                completion_message = "非常感谢您的配合，问诊已经完成。我已经为您整理了一份问诊摘要，稍后会交给医生进行专业诊断。祝您早日康复！"
                self.speak(completion_message)
                
                result["response"] = completion_message
                result["consultation_complete"] = True
            else:
                # Give feedback based on relevance
                if not self.response_relevant:
                    response_message = "您的回答可能与问题不太相关，让我们再试一次。"
                    self.speak(response_message)
                    result["response"] = response_message
                else:
                    result["response"] = "收到您的回答，继续下一个问题。"
            
            # Success
            result["success"] = True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result["error"] = f"Error during medical consultation: {e}"
            
        return result
    
    def run_consultation(self, exit_phrase: str = "结束问诊"):
        """Run the complete medical consultation
        
        Args:
            exit_phrase: Phrase to exit the consultation
            
        Returns:
            The consultation summary
        """
        print("Starting Medical Diagnosis Chatbot with logical flow control...")
        print(f"Say '{exit_phrase}' to end consultation.")
        
        # Start the listening thread
        self.listen_thread.start()
        
        # Initial greeting
        greeting = "您好，我是您的智能问诊助手。接下来我会通过提问来了解您的健康状况，这样能帮助医生更好地了解您的情况。准备好了吗？我们开始第一个问题。"
        self.speak(greeting)
        
        # Wait for a moment to let the greeting sink in
        time.sleep(1)
        
        running = True
        consultation_summary = None
        
        while running:
            # Run one interaction cycle
            result = self.run_once()
            
            if not result["success"]:
                if result["error"]:
                    print(f"Error: {result['error']}")
                    error_message = "抱歉，出现了一些技术问题。让我们继续问诊。"
                    self.speak(error_message)
            
            # Check if consultation is complete
            if result["consultation_complete"] and not consultation_summary:
                consultation_summary = self.generate_consultation_summary()
                print("\n" + "="*50)
                print("医疗问诊摘要:")
                print(consultation_summary)
                print("="*50 + "\n")
            
            # Check if user wants to exit
            if result["user_input"] and exit_phrase in result["user_input"]:
                goodbye = "感谢您的配合，问诊已结束。祝您健康！"
                self.speak(goodbye)
                running = False
            
            # Brief pause between interactions
            time.sleep(0.25)
        
        print("Medical consultation completed.")
        self.cleanup()
        
        # Return the consultation summary if available
        return consultation_summary if consultation_summary else self.generate_consultation_summary()
    
    def cleanup(self):
        """Clean up resources when shutting down"""
        self.listen_interrupt_stop.set()
    
    def is_response_relevant(self, response: str) -> bool:
        """Check if the user's response is relevant to the current question
        
        Args:
            response: The user's response text
            
        Returns:
            Boolean indicating if the response is relevant
        """
        current_question = self.get_base_question()
        
        # Create a system prompt for relevance check
        system_prompt = (
            "你是一个医疗问诊对话分析专家。你的任务是判断患者的回答是否与当前问题相关。"
            "请分析患者回答的内容是否针对了问题所问的方面，即使回答是'没有'也算相关。"
            "如果回答完全不相关或答非所问，请返回'不相关'，否则返回'相关'。"
        )
        
        # Create the relevance check prompt
        check_prompt = (
            f"当前问题：\"{current_question}\"\n\n"
            f"患者回答：\"{response}\"\n\n"
            f"请判断患者回答是否与当前问题相关？"
        )
        
        result = self.llm.generate_response(
            user_input=check_prompt,
            conversation_history=[{"role": "system", "content": system_prompt}]
        )
        
        if result["success"]:
            return "相关" in result["response"] and "不相关" not in result["response"]
        
        # Default to assuming the response is relevant if LLM call fails
        return True