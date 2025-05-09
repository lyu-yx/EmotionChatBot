"""
Emotion-Aware Chatbot
-------------------
Integrates ASR, LLM, TTS, and Emotion detection for a complete voice interaction experience.
"""

from turtle import listen
from typing import Optional, List, Dict, Any
import os
import time
import threading
import queue
import numpy as np
# Import our component interfaces directly from their modules
from src.asr.speech_recognition_engine import SpeechRecognizer, DashscopeSpeechRecognizer
from src.tts.speech_synthesis import TextToSpeech, StreamingTTSSynthesizer
from src.llm.language_model import LanguageModel, StreamingLanguageModel
from src.emotion.emotion_detector import EmotionDetector, DashscopeEmotionDetector, TextBasedEmotionDetector
from src.emotion.identify import EmotionDetectorCamera
from src.core.SharedQueue import SharedQueue as q
from src.core.SharedLock import SharedLock as lock
from datetime import datetime
import logging
import random
logging.basicConfig(level=logging.INFO)
from collections import deque
# Configuration flags
USE_TEXT_EMOTION_DETECTION = False  # Set to True to enable text-based emotion detection
USE_CAMERA_EMOTION_DETECTION = True  # Set to True to enable camera-based emotion detection


class EmotionAwareStreamingChatbot:
    """An emotion-aware streaming chatbot that integrates ASR, LLM, TTS, and emotion detection"""

    def __init__(self,
                 recognizer: Optional[SpeechRecognizer] = None,
                 tts: Optional[TextToSpeech] = None,
                 llm: Optional[LanguageModel] = None,
                 emotion_detector: Optional[EmotionDetector] = None,
                 system_prompt: Optional[str] = None,
                 language: str = "zh-cn",
                 use_text_emotion: bool = USE_TEXT_EMOTION_DETECTION,
                 use_camera_emotion: bool = USE_CAMERA_EMOTION_DETECTION,
                 camera_id: int = 0,
                 show_camera: bool = False):
        """Initialize the emotion-aware streaming chatbot

        Args:
            recognizer: Speech recognition engine (default: DashscopeSpeechRecognizer)
            tts: Text-to-speech engine (default: StreamingTTSSynthesizer)
            llm: Language model (default: StreamingLanguageModel)
            emotion_detector: Emotion detection component (default: DashscopeEmotionDetector)
            system_prompt: Optional system prompt to guide the LLM
            language: Language code for ASR (default: "zh-cn")
            use_text_emotion: Whether to use text-based emotion detection
            use_camera_emotion: Whether to use camera-based emotion detection
            camera_id: Camera ID to use for emotion detection
            show_camera: Whether to show camera feed
        """
        # Default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "你是一个有用的、能够理解情感的语音交互助手。"
                "请根据用户的情感状态，提供适合的回答。"
                "保持回应简短但信息丰富和有帮助。"
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

        # Text-based emotion detection configuration
        self.use_text_emotion = use_text_emotion
        if self.use_text_emotion:
            # Initialize emotion detector
            self.emotion_detector = emotion_detector if emotion_detector else DashscopeEmotionDetector()

            # Fall back to text-based emotion detector if Dashscope is not available
            if not getattr(self.emotion_detector, 'api_key', True):
                self.emotion_detector = TextBasedEmotionDetector()
                print("Warning: Using text-based emotion detector as fallback")
        else:
            # Dummy emotion detector
            self.emotion_detector = None
            print("Text-based emotion detection is disabled")

        # Camera-based emotion detection configuration
        self.use_camera_emotion = use_camera_emotion
        self.camera_detector = None

        if self.use_camera_emotion:
            try:
                # Check if we should use Chinese labels based on the language setting
                use_chinese = language.startswith("zh")

                # Define callback function for camera detection
                def on_camera_emotion_detected(result):
                    emotion = result["emotion"]
                    confidence = result["probability"]
                    self.camera_emotion = emotion
                    print(f"Camera detected emotion: {emotion} ({confidence*100:.1f}%)")

                # Initialize camera emotion detector
                self.camera_detector = EmotionDetectorCamera(
                    detection_interval=0.5,  # Check emotion every half second
                    use_chinese=use_chinese,
                    callback=on_camera_emotion_detected
                )

                # Start the camera detection
                success = self.camera_detector.start(camera_id=camera_id, show_video=show_camera)
                if success:
                    print(f"Camera-based emotion detection started (Camera ID: {camera_id})")
                else:
                    print("Failed to start camera-based emotion detection")
                    self.use_camera_emotion = False
            except Exception as e:
                print(f"Error initializing camera emotion detection: {e}")
                self.use_camera_emotion = False

        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []

        # Maximum number of conversation turns to keep in history
        self.max_history_length = 10

        # User emotion state tracking
        self.text_emotion = "neutral"
        self.camera_emotion = "neutral"

        # Buffer for collecting LLM output chunks
        self.text_buffer = ""

        # Sentence ending punctuation for splitting streaming text
        self.sentence_end_chars = ["。", "！", "？", ".", "!", "?", "；", ";"]

        # Flag to indicate if TTS is currently active
        self.is_speaking = False
        
        # Flag to judge whether the bot is active
        self.is_active = False
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Store the message
        self.queue = q()
        
        # Enhanced response cache with frequency tracking and TTL
        self.response_cache = {}
        self.cache_ttl = 3600  # Time-to-live for cache entries in seconds
        self.cache_hit_counter = {}  # Track frequency of cache hits
        self.cache_timestamp = {}  # Track when items were added to cache
        self.max_cache_size = 100  # Maximum number of items in cache
        
        # Barge-in control flags
        self.speech_should_stop = threading.Event()
        self.barge_in_sensitivity = 800  # Adjustable sensitivity (lower = more sensitive)
        self.barge_in_history = []  # Track recent audio levels for better detection
        self.barge_in_window_size = 5  # Number of samples to keep for barge-in detection
        
        # Audio level baseline for normalization
        self.audio_baseline = 400  # Initial estimate, will be adjusted
        self.audio_baseline_samples = []  # Samples for baseline calculation
        
        # Parallel processing threads and queues
        self.asr_queue = queue.Queue()  # For speech recognition results
        self.llm_queue = queue.Queue()  # For LLM processing results
        self.tts_queue = queue.Queue()  # For TTS processing
        
        # Priority queue for high-priority responses
        self.priority_tts_queue = queue.PriorityQueue(maxsize=5)
        
        # Prefetch settings
        self.prefetch_enabled = True
        self.preprocessing_complete = threading.Event()
        
        # Latency metrics
        self.latency_metrics = {
            "asr": [],
            "llm": [],
            "tts": [],
            "total": []
        }
        self.max_metrics_samples = 50
        
        # Control the listen_continuous thread
        self.listen_thread = threading.Thread(target=self.listen_continuous)
        self.listen_thread.daemon = True
        self.listen_interrupt_stop = threading.Event()
        
        # Processing thread for LLM
        self.llm_thread = threading.Thread(target=self.llm_process_continuous)
        self.llm_thread.daemon = True
        
        # Processing thread for TTS
        self.tts_thread = threading.Thread(target=self.tts_process_continuous)
        self.tts_thread.daemon = True
        
        # Lock to avoid listen confliction
        self.listen_lock = lock()
        
        # Try to warm up the models with simple queries
        # try:
        #     # Warm up LLM with a simple request
        #     self.llm.generate_response("Hello", [])
        #     print("LLM model warmed up")
        # except Exception as e:
        #     print(f"LLM warmup failed (not critical): {e}")
            
        # Dynamic threshold adjustment
        self.dynamic_threshold_enabled = True
            
        # Threads will be started in the run_continuous method
        print("Emotion-Aware Streaming Chatbot initialized")
    
    def listen_continuous(self):
        """Continuously listen for user input in a background thread"""
        while True:
            with self.listen_lock:
                if not self.listen_interrupt_stop.is_set():
                    try:
                        # # Check if speech is ongoing and should be interrupted
                        # if self.is_speaking:
                        #     # Check audio level to see if user is speaking
                        #     # If audio level is above threshold, interrupt current speech
                        #     audio_level = self._get_audio_level()
                        #     if (audio_level > 0.5):  # Adjust threshold as needed
                        #         print("User started speaking - interrupting current speech")
                        #         self.speech_should_stop.set()
                        #         with self.lock:
                        #             self.is_speaking = False
                        
                        # Recognize speech
                        result = self.recognizer.recognize_from_microphone()
                        if result and result["text"] != '':
                            # # Interrupt current speech when new input is detected
                            # if self.is_speaking:
                            #     print("New speech detected - interrupting current output")
                            #     self.speech_should_stop.set()
                            #     with self.lock:
                            #         self.is_speaking = False
                            
                            self.queue.put(result)
                            # Also put in ASR queue for parallel processing
                            self.asr_queue.put(result)
                    except Exception as e:
                        print(f"Listen thread exception: {e}")
            time.sleep(0.05)  # Small sleep to prevent CPU overuse
    
    def _get_audio_level(self):
        """Get current audio input level for barge-in detection"""
        try:
            import pyaudio
            import numpy as np
            
            # Use a short window to check audio level
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
            
            # Read a small sample to check level
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Calculate RMS as a measure of audio level
            rms = np.sqrt(np.mean(np.square(audio_data)))
            
            # Clean up
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            return rms
        except Exception as e:
            # Fall back to a default value if audio level detection fails
            print(f"Audio level detection failed: {e}")
            return 0
            
    def llm_process_continuous(self):
        """Process LLM requests in a separate thread with improved caching and prefetching"""
        import random  # For probabilistic preprocessing
        
        # Start preprocessing common responses
        if self.prefetch_enabled:
            preprocessing_thread = threading.Thread(
                target=self.preprocess_common_responses,
                daemon=True
            )
            preprocessing_thread.start()
        
        while True:
            try:
                # Get the next speech recognition result
                asr_result = self.asr_queue.get(timeout=0.5)
                
                # Skip empty results
                if not asr_result or not asr_result.get("text"):
                    continue
                
                user_input = asr_result["text"]
                start_time = time.time()
                
                # Update metrics for ASR (from recognition to LLM processing)
                if hasattr(asr_result, 'timestamp'):
                    asr_latency = start_time - asr_result.timestamp
                    self.latency_metrics["asr"].append(asr_latency)
                    self.latency_metrics["asr"] = self.latency_metrics["asr"][-self.max_metrics_samples:]
                
                # Check cache first for common responses
                cache_hit = False
                if user_input in self.response_cache:
                    cached_response = self.response_cache[user_input]
                    print(f"Cache hit: Using cached response for: {user_input}")
                    
                    # Update cache hit counter
                    if user_input in self.cache_hit_counter:
                        self.cache_hit_counter[user_input] += 1
                    else:
                        self.cache_hit_counter[user_input] = 1
                        
                    # Update timestamp to extend TTL
                    self.cache_timestamp[user_input] = time.time()
                    
                    # Put in high-priority queue for immediate response
                    self.priority_tts_queue.put((
                        1,  # Higher priority (lower number)
                        {
                            "input": user_input,
                            "response": cached_response,
                            "success": True,
                            "from_cache": True
                        }
                    ))
                    cache_hit = True
                
                # Process with LLM if not in cache
                if not cache_hit:
                    # Process emotion from text if enabled
                    if self.use_text_emotion:
                        current_emotion = self.process_emotion(user_input)
                    
                    # Get current emotion
                    if self.use_camera_emotion:
                        current_emotion = self.get_current_emotion()
                    
                    # Add emotion context to the user input
                    emotion_context = f"[User emotion: {current_emotion}] "
                    augmented_input = emotion_context + user_input
                    
                    # Start processing with LLM
                    print(f"LLM thread processing: '{user_input}' (Emotion: {current_emotion})")
                    
                    llm_result = self.llm.generate_response(
                        user_input=augmented_input,
                        conversation_history=self.conversation_history
                    )
                    
                    # Update LLM latency metrics
                    llm_time = time.time() - start_time
                    self.latency_metrics["llm"].append(llm_time)
                    self.latency_metrics["llm"] = self.latency_metrics["llm"][-self.max_metrics_samples:]
                    
                    # Cache frequently used responses (only if appropriate)
                    if llm_result["success"]:
                        if len(user_input) < 25 and len(llm_result["response"]) < 150:
                            # Check if this is a common question worth caching
                            if random.random() < 0.8 or len(user_input) < 10:  # Higher chance for short inputs
                                self.response_cache[user_input] = llm_result["response"]
                                self.cache_timestamp[user_input] = time.time()
                                self.cache_hit_counter[user_input] = 1  # Initialize counter
                                print(f"Cached new response for: {user_input}")
                        
                        # Update conversation history
                        self.conversation_history.append({"role": "user", "content": user_input})
                        self.conversation_history.append({"role": "assistant", "content": llm_result["response"]})
                        
                        # Trim history if it exceeds maximum length
                        if len(self.conversation_history) > self.max_history_length * 2:
                            self.conversation_history = self.conversation_history[-self.max_history_length*2:]
                    
                    # Regular priority for non-cached responses
                    self.llm_queue.put({
                        "input": user_input,
                        "response": llm_result["response"],
                        "success": llm_result["success"],
                        "error": llm_result.get("error"),
                        "emotion": current_emotion,
                        "from_cache": False
                    })
                
                # Update total latency metrics (from recognition to LLM completion)
                total_time = time.time() - start_time
                self.latency_metrics["total"].append(total_time)
                self.latency_metrics["total"] = self.latency_metrics["total"][-self.max_metrics_samples:]
                
                # Print performance insights occasionally
                if random.random() < 0.1:  # 10% chance
                    self._print_performance_insights()
                
            except queue.Empty:
                # Queue timeout, continue waiting
                pass
            except Exception as e:
                print(f"LLM thread exception: {e}")
                
            time.sleep(0.05)  # Small sleep to prevent CPU overuse
    
    def tts_process_continuous(self):
        """Process TTS requests in a separate thread with enhanced interruption handling"""
        while True:
            try:
                # Check priority queue first (for interruptions and urgent responses)
                try:
                    # Non-blocking check of priority queue (timeout=0.01)
                    priority, llm_result = self.priority_tts_queue.get(timeout=0.01)
                    print(f"Processing priority TTS request (priority: {priority})")
                except queue.Empty:
                    # If no priority items, get from regular queue
                    llm_result = self.llm_queue.get(timeout=0.5)
                
                # Skip failed results
                if not llm_result["success"]:
                    print(f"Skipping TTS for failed LLM result: {llm_result.get('error')}")
                    continue
                print("Success get llm result")
                response_text = llm_result["response"]
                from_cache = llm_result.get("from_cache", False)
                
                # Split into sentences for faster response
                sentences = self._split_into_sentences(response_text)
                
                # Prepare first sentence immediately for faster response
                if sentences and not from_cache:
                    first_sentence = sentences[0]
                    try:
                        # Prepare synthesis but don't play yet
                        self.tts.prepare_synthesis(first_sentence)
                    except Exception as e:
                        print(f"TTS preparation exception: {e}")
                
                # Process each sentence with interruption handling
                for i, sentence in enumerate(sentences):
                    # Check if we should stop speaking before starting next sentence
                    # if self.speech_should_stop.is_set():
                    #     print("TTS interrupted - stopping speech")
                    #     self.speech_should_stop.clear()
                    #     break
                    
                    # Set speaking flag
                    with self.lock:
                        self.is_speaking = True
                    
                    start_time = time.time()
                    
                    # Speak the sentence
                    try:
                        # Track if this is cached content for metrics
                        is_cached = from_cache or (i > 0)  # First sentence might be prepared
                        
                        # Actually speak the prepared sentence or a new one
                        if i == 0 and not from_cache:
                            # Use the prepared sentence
                            self.tts.play_prepared()
                        else:
                            # Synthesize and play directly
                            self.tts.speak(sentence)
                        
                        # Record TTS latency for monitoring
                        tts_time = time.time() - start_time
                        self.latency_metrics["tts"].append(tts_time)
                        self.latency_metrics["tts"] = self.latency_metrics["tts"][-self.max_metrics_samples:]
                        
                    except Exception as e:
                        print(f"TTS exception: {e}")
                    
                    # Prepare next sentence while speaking current one (if not last)
                    if i < len(sentences) - 1 and not self.speech_should_stop.is_set():
                        try:
                            next_sentence = sentences[i + 1]
                            self.tts.prepare_synthesis(next_sentence)
                        except Exception as e:
                            print(f"TTS preparation exception: {e}")
                    
                    # Clear speaking flag after each sentence
                    with self.lock:
                        self.is_speaking = False
                    
                    # Small pause between sentences for natural speech rhythm
                    # Adjust pause based on punctuation
                    if i < len(sentences) - 1:  # Don't pause after last sentence
                        if sentence.endswith((".", "!", "?")):
                            time.sleep(0.15)  # Longer pause for sentence boundaries
                        elif sentence.endswith((",", ";", ":")):
                            time.sleep(0.1)  # Medium pause for clause boundaries
                        else:
                            time.sleep(0.05)  # Short pause otherwise
                
                # Run cache management periodically
                if random.random() < 0.05:  # 5% chance on each response
                    threading.Thread(target=self.manage_cache, daemon=True).start()
                
            except queue.Empty:
                # Queue timeout, continue waiting
                pass
            except Exception as e:
                print(f"TTS thread exception: {e}")
                # Make sure to reset speaking flag on error
                with self.lock:
                    self.is_speaking = False
                    
            time.sleep(0.02)  # Small sleep to prevent CPU overuse
    
    def _split_into_sentences(self, text):
        """Split text into sentences for better streaming"""
        sentences = []
        current = ""
        
        # Process character by character
        for char in text:
            current += char
            
            # Check if we've reached a sentence boundary
            if char in self.sentence_end_chars:
                sentences.append(current)
                current = ""
        
        # Add any remaining text
        if current:
            sentences.append(current)
            
        return sentences
    

        #Flag to judge whether the bot is active
        self.is_active = False

        # Lock for thread safety
        self.lock = threading.Lock()

        #Store the message
        self.queue = q()

        #Control the listen_continuous thread
        self.listen_all = threading.Thread(target = self.listen_continuous)
        self.listen_all.daemon = True
        self.listen_interrupt_stop = threading.Event()

        #Lock to avoid listen confliction
        self.listen_lock = lock()

        # Emotion monitoring thread
        self._emotion_monitor_active = False
        self._emotion_monitor_thread = None
        self._stop_emotion_monitor = threading.Event()
        self.emotion_window = deque(maxlen=10)  # 10秒滑动窗口
        self.negative_threshold = 0.5
        self._passive_running = False

        print("Emotion-Aware Streaming Chatbot initialized")

    def start_emotion_monitoring(self):
        """Start the background emotion monitoring thread"""
        if self._emotion_monitor_thread is None or not self._emotion_monitor_thread.is_alive():
            self._stop_emotion_monitor.clear()
            self._emotion_monitor_thread = threading.Thread(
                target=self._emotion_monitor_loop,
                daemon=True
            )
            self._emotion_monitor_thread.start()
            print("Started background emotion monitoring")

    def stop_emotion_monitoring(self):
        """Stop the background emotion monitoring thread"""
        self._stop_emotion_monitor.set()
        if self._emotion_monitor_thread is not None:
            self._emotion_monitor_thread.join(timeout=1)
            print("Stopped background emotion monitoring")

    def _emotion_monitor_loop(self):
        while not self._stop_emotion_monitor.is_set():
            # 获取当前情绪（假设返回字典包含emotion和confidence）
            result = self.camera_detector.get_latest_emotion()

            # 记录情绪状态（1=负面, 0=非负面）
            is_negative = 1 if result["emotion"] in ["sad"] else 0
            self.emotion_window.append(is_negative)

            # 每0.5秒检测一次（10秒窗口=20次检测）
            #time.sleep(0.5)

            # 当窗口满时计算负面情绪占比
            if len(self.emotion_window) == self.emotion_window.maxlen:
                negative_ratio = sum(self.emotion_window) / len(self.emotion_window)

                # 达到阈值且当前仍处于负面状态
                if (negative_ratio >= self.negative_threshold and
                        is_negative == 1):
                    self._trigger_comfort_behavior()

    def _trigger_comfort_behavior(self):
        """触发安慰行为前的最终校验"""
        # 1. 确保不在对话状态
        if self.is_active or self.is_speaking:
            return

        # 2. 二次验证当前情绪
        current_emotion = self.camera_detector.get_latest_emotion()
        if current_emotion["probability"] < 0.7:
            return

        # 3. 执行安慰（讲笑话/音乐等）
        self._tell_joke()

        # 4. 清空窗口避免重复触发
        self.emotion_window.clear()

    def _tell_joke(self):
        """专用讲笑话方法"""
        if self.is_speaking:  # 不打断现有语音
            return
        try:
            self.speak(f"检测到您似乎心情不太好，让我讲个笑话吧！")

            # 2. 专门生成笑话的prompt
            joke_prompt = "检测到用户情绪不好，讲个笑话吧"

            # 3. 获取笑话响应
            collected_response = ""

            def process_chunk(chunk):
                nonlocal collected_response
                collected_response += chunk
                print(".", end="", flush=True)

            joke_result = self.llm.generate_stream_response(
                user_input=joke_prompt,  # 使用专门的笑话prompt
                conversation_history=self.conversation_history,
                chunk_callback=process_chunk
            )

            # 4. 提取文本内容
            if joke_result.get("success"):
                joke_text = joke_result.get("response", "我想不出笑话了...")
                print(f"\n生成的笑话：{joke_text}")
                self.speak(joke_text)  # 只朗读笑话内容
            else:
                error_msg = joke_result.get("error", "生成笑话失败")
                print(f"笑话生成失败：{error_msg}")
                self.speak("哎呀，暂时想不出笑话了")

        except Exception as e:
            print(f"讲笑话失败: {e}")
            self.speak("哎呀，我的笑话库卡住了")

    def listen_continuous(self):
        while True:
            with self.listen_lock:
                if not self.listen_interrupt_stop.is_set():
                    try :
                        result = self.recognizer.recognize_from_microphone()
                        if result and result["text"] != '':
                            self.queue.put(result)
                    except Exception as e:
                        print(f"监听线程异常：{e}")
            time.sleep(0.3)

    def get_current_emotion(self) -> str:
        """Get the current emotion based on enabled detection methods

        Returns:
            Current detected emotion
        """
        # If both methods are disabled, return neutral
        if not self.use_text_emotion and not self.use_camera_emotion:
            return "neutral"

        # If only text emotion is enabled
        if self.use_text_emotion and not self.use_camera_emotion:
            return self.text_emotion

        # If only camera emotion is enabled
        if not self.use_text_emotion and self.use_camera_emotion:
            return self.camera_emotion

        # If both are enabled, prefer camera emotion if it has high confidence
        if self.use_camera_emotion:
            camera_result = self.camera_detector.get_latest_emotion()
            confidence = camera_result["probability"]
            # Use camera emotion if confidence is high enough
            if confidence > 0.6:
                return self.camera_emotion

        # Fall back to text emotion or neutral
        return self.text_emotion if self.use_text_emotion else "neutral"

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

    def process_emotion(self, text: str) -> str:
        """Process text to detect emotions and update user emotional state

        Args:
            text: User's text input to analyze

        Returns:
            Detected dominant emotion
        """
        # Skip emotion detection if disabled
        if not self.use_text_emotion:
            return "neutral"

        # Detect emotion
        emotion_result = self.emotion_detector.detect_emotion_from_text(text)

        if emotion_result["success"]:
            # Update the user's emotional state
            self.text_emotion = emotion_result["dominant_emotion"]

            # Log the detected emotion
            print(f"Detected text emotion: {self.text_emotion} "
                  f"(confidence: {emotion_result['emotions'].get(self.text_emotion, 0):.2f})")
        else:
            print(f"Emotion detection failed: {emotion_result.get('error', 'Unknown error')}")

        return self.text_emotion

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
            print("after speak")
            # Add a small delay after speaking to avoid cutting off
            #time.sleep(0.1)
            
            return result
        finally:
            # Make sure to reset the flag even if an error occurs
            with self.lock:
                self.is_speaking = False

    def process_streaming(self, user_input: str, emotion: str = "neutral", full_response: bool = False) -> Dict[str, Any]:
        """Process user input with streaming LLM and TTS, taking emotion into account

        Args:
            user_input: User's text input
            emotion: Detected emotion of the user
            full_response: If True, collects the entire response before speaking instead of speaking in chunks

        Returns:
            Dict with response generation results
        """
        start = time.time()
        result = {
            "user_input": user_input,
            "response": "",
            "success": False,
            "error": None,
            "user_emotion": emotion
        }

        try:
            print(f"Processing: '{user_input}' (Emotion: {emotion})")

            # Add emotion context to the user input
            emotion_context = f"[User emotion: {emotion}] "
            augmented_input = emotion_context + user_input

            if full_response:
                # For full response mode, collect entire text before speaking
                collected_response = ""

                # Define the callback function for handling text chunks
                def process_chunk(chunk):
                    nonlocal collected_response
                    # Add to the complete response
                    collected_response += chunk
                    # Print progress indicator
                    print(".", end="", flush=True)

                # Call the language model with streaming enabled
                llm_result = self.llm.generate_stream_response(
                    user_input=augmented_input,
                    conversation_history=self.conversation_history,
                    chunk_callback=process_chunk
                )

                # Speak the full collected response
                if collected_response:
                    print("\nSpeaking full response...")
                    self.speak(collected_response)
            else:
                # Define the callback function for handling text chunks (original behavior)
                def process_chunk(chunk):
                    # Add to the complete response
                    self.text_buffer += chunk

                    # Check if we have a complete sentence or enough text
                    if self._should_synthesize():
                        text_to_synthesize = self.text_buffer
                        self.text_buffer = ""
                        # Synthesize and play this chunk with speaking protection
                        self.speak(text_to_synthesize)

                # Call the language model with streaming enabled
                llm_result = self.llm.generate_stream_response(
                    user_input=augmented_input,
                    conversation_history=self.conversation_history,
                    chunk_callback=process_chunk
                )

                # Process any remaining text in the buffer
                if self.text_buffer:
                    self.speak(self.text_buffer)
                    self.text_buffer = ""

            # Update the result
            result["response"] = llm_result["response"]
            result["success"] = llm_result["success"]
            result["error"] = llm_result["error"]

            # Update conversation history if successful
            if llm_result["success"]:
                # Add user message (without the emotion context)
                self.conversation_history.append({"role": "user", "content": user_input})

                # Add assistant response
                self.conversation_history.append({"role": "assistant", "content": llm_result["response"]})

                # Trim history if it exceeds maximum length
                if len(self.conversation_history) > self.max_history_length * 2:
                    # Keep only the most recent turns
                    self.conversation_history = self.conversation_history[-self.max_history_length*2:]

        except Exception as e:
            result["error"] = f"Error during streaming processing: {e}"

        return result

    def _should_synthesize(self, min_chunk_size=20):
        """Determine if the current buffer should be synthesized

        Check if the text buffer contains a complete sentence or is long enough

        Args:
            min_chunk_size: Minimum size to trigger synthesis

        Returns:
            Boolean indicating if synthesis should occur
        """
        # Check for sentence ending punctuation
        for char in self.sentence_end_chars:
            if char in self.text_buffer:
                return True

        # Check if buffer is long enough
        if len(self.text_buffer) >= min_chunk_size:
            return True

        return False
    
    def run_once(self, full_response: bool = False, exit_phase: str = "再见") -> Dict[str, Any]:
        """Run one complete interaction cycle

        Args:
            full_response: If True, wait for the complete response before speaking

        Returns:
            Dict with interaction results
        """
        result = {
            "user_input": "",
            "response": "",
            "success": False,
            "error": None,
            "user_emotion": "neutral",
            "exit": False
        }

        try:
            # Ensure we're not speaking before listening
            while True:
                with self.lock:
                    if not self.is_speaking:
                        break
                print("Waiting for speech to complete before listening...")
                time.sleep(0.25)
                
            # Step 1: Listen for user input
            print("Listening for user input...")
            listen_result = self.queue.get()
            timestamp = datetime.now().timestamp()
            logging.info(f"after queue get:{timestamp}")
            if not listen_result["success"]:
                result["error"] = listen_result["error"]
                return result
            if exit_phase in listen_result["text"]:
                result["exit"] = True
                return result
            user_input = listen_result["text"]
            result["user_input"] = user_input

            # Step 2: Process emotion from text (if enabled)
            if self.use_text_emotion:
                self.process_emotion(user_input)

            # Step 3: Get the current emotion (combines camera and/or text)
            current_emotion = self.get_current_emotion()
            result["user_emotion"] = current_emotion
            # Step 4: Process with streaming LLM and TTS
            process_result = self.process_streaming(user_input, current_emotion, full_response)

            if not process_result["success"]:
                result["error"] = process_result["error"]
                return result

            result["response"] = process_result["response"]

            # All steps succeeded
            result["success"] = True

        except Exception as e:
            result["error"] = f"Error during interaction: {e}"

        return result

    def run_continuous(self, wake_word: Optional[str] = None, exit_phrase: str = "exit", full_response: bool = False, activation_timeout: int = 60, debug_mode: bool = True):
        """Run the chatbot in continuous mode with improved parallel processing"""
        import random  # For audio calibration
        
        language_display = "Chinese" if self.language.startswith("zh") else "English"
        self.start_emotion_monitoring()
        # Prepare emotion detection mode for display
        emotion_modes = []
        if self.use_text_emotion:
            emotion_modes.append("Text")
        if self.use_camera_emotion:
            emotion_modes.append("Camera")

        emotion_mode_display = "+".join(emotion_modes) if emotion_modes else "Disabled"

        print(f"Emotion-Aware Chatbot started ({language_display} mode)")
        print(f"Emotion detection: {emotion_mode_display}")

        # Wake word configuration
        using_wake_word = wake_word is not None and len(wake_word.strip()) > 0
        if using_wake_word:
            print(f"Wake word: '{wake_word}' (Activation timeout: {activation_timeout} seconds)")
            print(f"Debug mode: {'Enabled' if debug_mode else 'Disabled'}")
            if self.language.startswith("zh"):
                waiting_message = f"等待唤醒词 '{wake_word}'..."
                activation_message = "已激活! 请说出您的问题..."
                timeout_message = "已超时，进入休眠模式..."
            else:
                waiting_message = f"Waiting for wake word '{wake_word}'..."
                activation_message = "Activated! Please speak your question..."
                timeout_message = "Timeout, entering sleep mode..."
        else:
            print("Wake word: Disabled (Always active)")

        print(f"Say '{exit_phrase}' to exit.")
        speech_mode = "full response mode" if full_response else "streaming mode"
        print(f"Speech output using {speech_mode}")

        # Initial greeting
        if language_display == "Chinese":
            if using_wake_word:
                greeting = f"你好! 我是能够感知情绪的语音助手。当你需要我时，请说'{wake_word}'来唤醒我。"
            else:
                greeting = "你好! 我是能够感知情绪的语音助手。请问今天我能帮您什么？"
        else:
            if using_wake_word:
                greeting = f"Hello! I'm an emotion-aware voice assistant. Say '{wake_word}' to wake me up when you need me."
            else:
                greeting = "Hello! I'm an emotion-aware voice assistant. How can I help you today?"
        
        print("Starting initial greeting...") 
        self.speak(greeting)

        # Start the emotion monitoring thread
        self.start_emotion_monitoring()

        # Keep track of activation state when using wake word
        self.is_active = not using_wake_word  # If not using wake word, always active
        active_until = 0  # Timestamp when activation expires
        running = True
        
        # Start all background threads
        print("Starting background threads...")
        self.listen_thread.start()
        # self.llm_thread.start()
        # self.tts_thread.start()
        
        # Periodically recalibrate audio baseline
        last_calibration = time.time()
        
        while running:
            print("\n" + "="*50)
            
            # Periodically recalibrate audio baseline (every 30 minutes)
            if time.time() - last_calibration > 1800:  # 30 minutes
                print("Recalibrating audio baseline...")
                # This will happen in the audio monitor thread
                self.audio_baseline_samples = []
                last_calibration = time.time()
            
            # Check if we need to listen for wake word
            if using_wake_word and not self.is_active:
                print(waiting_message)

                # Wait for wake word
                wake_word_detected = False
                while not wake_word_detected and running:
                    try:
                        # Listen for wake word
                        listen_result = self.queue.get()
                        if listen_result["success"]:
                            text = listen_result["text"].lower()

                            # Debug output - show what's being recognized
                            if debug_mode:
                                print(f"DEBUG - Wake word phase recognized: '{text}'")
                                print(f"DEBUG - Looking for wake word: '{wake_word.lower()}'")
                                print(f"DEBUG - Wake word in text: {wake_word.lower() in text}")
                                similarity = self._calculate_text_similarity(wake_word.lower(), text)
                                print(f"DEBUG - Text similarity score: {similarity:.2f}")

                            # Check if user said the exit phrase
                            if text == exit_phrase.lower() or exit_phrase.lower() in text:
                                if language_display == "Chinese":
                                    goodbye = "再见! 祝您有美好的一天。"
                                else:
                                    goodbye = "Goodbye! Have a great day."
                                self.speak(goodbye)
                                running = False
                                break

                            # Check if user said the wake word
                            if wake_word.lower() in text:
                                print(f"Wake word detected: '{text}'")
                                wake_word_detected = True
                                self.is_active = True
                                active_until = time.time() + activation_timeout if activation_timeout > 0 else float('inf')
                                print(activation_message)

                                # Acknowledge activation with a sound or short response
                                if language_display == "Chinese":
                                    self.speak("我在听，请说。")
                                else:
                                    self.speak("I'm listening, go ahead.")

                                # Give user a moment to start their question
                                time.sleep(0.25)
                        
                        # Brief pause between wake word detection attempts to reduce CPU usage
                        time.sleep(0.1)

                    except KeyboardInterrupt:
                        print("\nExiting due to keyboard interrupt...")
                        running = False
                        break

                # If we're exiting the loop and not because of wake word detection, continue to next loop iteration
                if not wake_word_detected:
                    continue

            # If we're active, check if activation has timed out
            if using_wake_word and self.is_active and activation_timeout > 0 and time.time() > active_until:
                print(timeout_message)
                self.is_active = False
                continue

            # Regular active mode
            print("Listening for your command...")
            print("Please speak now...")

            # Run one interaction cycle
            result = self.run_once(full_response, exit_phrase)
            
            # Print recognition result for debugging
            if result["success"]:
                print(f"\nYou said: {result['user_input']} (Emotion: {result['user_emotion']})")
                print(f"Response: {result['response']}")

                # If using wake word, extend the activation period after each successful interaction
                if using_wake_word and activation_timeout > 0:
                    active_until = time.time() + activation_timeout
                    print(f"Activation extended for {activation_timeout} seconds")

            # Check for exit phrase
            if result["exit"] == True:
                if language_display == "Chinese":
                    goodbye = "再见! 祝您有美好的一天。"
                else:
                    goodbye = "Goodbye! Have a great day."
                self.speak(goodbye)
                running = False
                self.is_active = False
                print("Exiting chatbot...")

            # If there was an error, report it
            elif not result["success"] and result["error"]:
                print("No message")
            
            # Brief pause between interactions
            time.sleep(0.25)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity score between 0 and 1
        """
        # Simple contains check
        if text1 in text2:
            return 1.0

        # Check for partial matches
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def cleanup(self):
        """Clean up resources when shutting down"""
        # Stop emotion monitoring
        self.stop_emotion_monitoring()

        # Stop camera detector if it was initialized
        if self.use_camera_emotion and self.camera_detector is not None:
            try:
                self.camera_detector.stop()
                print("Camera emotion detection stopped")
            except Exception as e:
                print(f"Error stopping camera detection: {e}")
    
    def audio_level_monitor(self):
        """Continuously monitor audio levels for improved barge-in detection"""
        try:
            import pyaudio
            import numpy as np
            import time
            
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            
            # Audio parameters
            CHUNK = 512  # Smaller chunks for more frequent checks
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            
            # Open stream in non-blocking mode
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                start=True,
                input_device_index=None,
                stream_callback=None
            )
            
            # Calibration phase
            print("Calibrating audio levels for barge-in detection...")
            self.audio_baseline_samples = []
            
            # Collect baseline samples while not speaking
            calibration_time = 2  # seconds
            start_time = time.time()
            while time.time() - start_time < calibration_time:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    rms = np.sqrt(np.mean(np.square(audio_data)))
                    self.audio_baseline_samples.append(rms)
                    time.sleep(0.01)  # Small sleep to prevent overwhelming the CPU
                except Exception as e:
                    print(f"Calibration sample error: {e}")
            
            # Calculate baseline from collected samples
            if self.audio_baseline_samples:
                # Use a percentile instead of mean to be robust against outliers
                self.audio_baseline = np.percentile(self.audio_baseline_samples, 75)
                print(f"Audio baseline level calibrated: {self.audio_baseline:.2f}")
            
            # Main monitoring loop
            while True:
                if self.is_speaking:
                    try:
                        # Check audio level
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        
                        # Calculate RMS
                        rms = np.sqrt(np.mean(np.square(audio_data)))
                        
                        # Add to history
                        self.barge_in_history.append(rms)
                        # Keep history at fixed size
                        if len(self.barge_in_history) > self.barge_in_window_size:
                            self.barge_in_history.pop(0)
                        
                        # Dynamic threshold adjustment if enabled
                        if self.dynamic_threshold_enabled and len(self.barge_in_history) >= 3:
                            # Calculate average of recent levels
                            recent_avg = sum(self.barge_in_history[-3:]) / 3
                            
                            # Check if current level is significantly higher (potential speech)
                            threshold = self.barge_in_sensitivity
                            
                            # Adjust threshold based on speaking volume
                            if recent_avg > self.audio_baseline * 2:
                                # Higher threshold during louder output
                                threshold = self.barge_in_sensitivity * 1.5
                            
                            # Trigger barge-in if threshold exceeded
                            if rms > threshold:
                                # Additional check: must exceed baseline by significant amount
                                if rms > self.audio_baseline * 3:
                                    print(f"Barge-in detected (Level: {rms:.2f}, Threshold: {threshold:.2f})")
                                    self.speech_should_stop.set()
                                    with self.lock:
                                        self.is_speaking = False
                    except Exception as e:
                        if "Input overflowed" not in str(e):  # Ignore common overflow errors
                            print(f"Audio monitoring error: {e}")
                
                time.sleep(0.02)  # Check frequently but don't overwhelm CPU
                
        except Exception as e:
            print(f"Audio monitor thread error: {e}")
        finally:
            # Clean up resources
            if 'stream' in locals() and stream:
                stream.stop_stream()
                stream.close()
            if 'p' in locals() and p:
                p.terminate()
    
    def manage_cache(self):
        """Clean up and optimize the response cache"""
        try:
            current_time = time.time()
            
            # Items to remove
            to_remove = []
            
            # Check each item in cache
            for key in self.response_cache:
                # Check if TTL expired
                if key in self.cache_timestamp:
                    age = current_time - self.cache_timestamp[key]
                    if age > self.cache_ttl:
                        to_remove.append(key)
            
            # Remove expired items
            for key in to_remove:
                del self.response_cache[key]
                if key in self.cache_timestamp:
                    del self.cache_timestamp[key]
                if key in self.cache_hit_counter:
                    del self.cache_hit_counter[key]
            
            # If cache is still too large, remove least frequently used items
            if len(self.response_cache) > self.max_cache_size:
                # Sort by hit count (ascending)
                sorted_items = sorted(
                    self.cache_hit_counter.items(),
                    key=lambda x: x[1]
                )
                
                # Remove least used items until we're under the limit
                items_to_remove = len(self.response_cache) - self.max_cache_size
                for i in range(items_to_remove):
                    if i < len(sorted_items):
                        key = sorted_items[i][0]
                        if key in self.response_cache:
                            del self.response_cache[key]
                            if key in self.cache_timestamp:
                                del self.cache_timestamp[key]
                            if key in self.cache_hit_counter:
                                del self.cache_hit_counter[key]
            
            if to_remove:
                print(f"Cache cleanup: removed {len(to_remove)} expired items")
                
        except Exception as e:
            print(f"Cache management error: {e}")
    
    def preprocess_common_responses(self):
        """Preprocess common responses for faster retrieval"""
        if not self.prefetch_enabled:
            return
        
        common_phrases = []
        
        # Language-specific common phrases
        if self.language.startswith("zh"):
            common_phrases = [
                "你好",
                "谢谢",
                "再见",
                "是的",
                "不是",
                "我不知道",
                "请帮我",
                "什么时候",
                "现在几点",
                "今天天气怎么样"
            ]
        else:
            common_phrases = [
                "hello",
                "thank you",
                "goodbye",
                "yes",
                "no",
                "I don't know",
                "please help me",
                "what time is it",
                "what's the weather today",
                "how are you"
            ]
        
        print("Preprocessing common responses...")
        
        # Process common phrases in background
        for phrase in common_phrases:
            if phrase not in self.response_cache:
                try:
                    # Generate response for common phrase
                    result = self.llm.generate_response(
                        user_input=phrase,
                        conversation_history=[]
                    )
                    
                    if result["success"]:
                        # Add to cache
                        self.response_cache[phrase] = result["response"]
                        self.cache_timestamp[phrase] = time.time()
                        self.cache_hit_counter[phrase] = 0
                        print(f"Preprocessed: '{phrase}'")
                except Exception as e:
                    print(f"Error preprocessing '{phrase}': {e}")
        
        self.preprocessing_complete.set()
        print("Common response preprocessing completed")
    
    def _print_performance_insights(self):
        """Print performance metrics for monitoring"""
        try:
            metrics = []
            
            # Only print metrics that have data
            for name, values in self.latency_metrics.items():
                if values:
                    avg = sum(values) / len(values)
                    metrics.append(f"{name}={avg:.3f}s")
            
            if metrics:
                print(f"Performance metrics (avg): {', '.join(metrics)}")
                
            # Print cache stats
            cache_size = len(self.response_cache)
            print(f"Response cache: {cache_size} items")
            
        except Exception as e:
            print(f"Error printing performance insights: {e}")