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
logging.basicConfig(level=logging.INFO)
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
        
        # Response cache for common phrases
        self.response_cache = {}
        
        # Barge-in control flags
        self.speech_should_stop = threading.Event()
        
        # Parallel processing threads and queues
        self.asr_queue = queue.Queue()  # For speech recognition results
        self.llm_queue = queue.Queue()  # For LLM processing results
        self.tts_queue = queue.Queue()  # For TTS processing
        
        # Control the listen_continuous thread
        self.listen_all = threading.Thread(target=self.listen_continuous)
        self.listen_all.daemon = True
        self.listen_interrupt_stop = threading.Event()
        
        # Processing thread for LLM
        self.llm_thread = threading.Thread(target=self.llm_process_continuous)
        self.llm_thread.daemon = True
        
        # Processing thread for TTS
        self.tts_thread = threading.Thread(target=self.tts_process_continuous)
        self.tts_thread.daemon = True
        
        # Lock to avoid listen confliction
        self.listen_lock = lock()
        
        # Warmup the models by initializing with empty calls
        self._warmup_models()
        
        print("Emotion-Aware Streaming Chatbot initialized")
    
    def _warmup_models(self):
        """Warm up models with dummy requests to reduce initial latency"""
        try:
            # Warm up LLM with a simple request
            self.llm.generate_response("Hello", [])
            
            # Warm up TTS with a simple request (without actually playing)
            self.tts.prepare_synthesis("Hello")
            
            print("Models warmed up to reduce initial latency")
        except Exception as e:
            print(f"Model warmup failed (this is not critical): {e}")
            
    def listen_continuous(self):
        """Continuously listen for user input in a background thread"""
        while True:
            with self.listen_lock:
                if not self.listen_interrupt_stop.is_set():
                    try:
                        # Check if speech is ongoing and should be interrupted
                        if self.is_speaking:
                            # Check audio level to see if user is speaking
                            # If audio level is above threshold, interrupt current speech
                            audio_level = self._get_audio_level()
                            if audio_level > 1000:  # Adjust threshold as needed
                                print("User started speaking - interrupting current speech")
                                self.speech_should_stop.set()
                                with self.lock:
                                    self.is_speaking = False
                        
                        # Recognize speech
                        result = self.recognizer.recognize_from_microphone()
                        if result and result["text"] != '':
                            # Interrupt current speech when new input is detected
                            if self.is_speaking:
                                print("New speech detected - interrupting current output")
                                self.speech_should_stop.set()
                                with self.lock:
                                    self.is_speaking = False
                            
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
        """Process LLM requests in a separate thread"""
        while True:
            try:
                # Get the next speech recognition result
                asr_result = self.asr_queue.get(timeout=0.5)
                
                # Skip empty results
                if not asr_result or not asr_result.get("text"):
                    continue
                
                user_input = asr_result["text"]
                
                # Check cache first for common responses
                if user_input in self.response_cache:
                    cached_response = self.response_cache[user_input]
                    print(f"Using cached response for: {user_input}")
                    self.llm_queue.put({
                        "input": user_input,
                        "response": cached_response,
                        "success": True,
                        "from_cache": True
                    })
                    continue
                
                # Process emotion from text if enabled
                if self.use_text_emotion:
                    self.process_emotion(user_input)
                
                # Get current emotion
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
                
                # Put the result in the queue for TTS processing
                if llm_result["success"]:
                    # Cache frequently used responses (only if they're short enough)
                    if len(user_input) < 20 and len(llm_result["response"]) < 100:
                        self.response_cache[user_input] = llm_result["response"]
                    
                    # Update conversation history
                    self.conversation_history.append({"role": "user", "content": user_input})
                    self.conversation_history.append({"role": "assistant", "content": llm_result["response"]})
                    
                    # Trim history if it exceeds maximum length
                    if len(self.conversation_history) > self.max_history_length * 2:
                        self.conversation_history = self.conversation_history[-self.max_history_length*2:]
                
                self.llm_queue.put({
                    "input": user_input,
                    "response": llm_result["response"],
                    "success": llm_result["success"],
                    "error": llm_result.get("error"),
                    "emotion": current_emotion,
                    "from_cache": False
                })
                
            except queue.Empty:
                # Queue timeout, continue waiting
                pass
            except Exception as e:
                print(f"LLM thread exception: {e}")
                
            time.sleep(0.05)  # Small sleep to prevent CPU overuse
    
    def tts_process_continuous(self):
        """Process TTS requests in a separate thread"""
        while True:
            try:
                # Get the next LLM result
                llm_result = self.llm_queue.get(timeout=0.5)
                
                # Skip failed results
                if not llm_result["success"]:
                    print(f"Skipping TTS for failed LLM result: {llm_result.get('error')}")
                    continue
                
                response_text = llm_result["response"]
                
                # Split into sentences for faster response
                sentences = self._split_into_sentences(response_text)
                
                # Process each sentence
                for i, sentence in enumerate(sentences):
                    # Check if we should stop speaking
                    if self.speech_should_stop.is_set():
                        print("TTS interrupted - stopping speech")
                        self.speech_should_stop.clear()
                        break
                    
                    print(f"TTS processing sentence {i+1}/{len(sentences)}")
                    
                    # Set speaking flag
                    with self.lock:
                        self.is_speaking = True
                    
                    # Speak the sentence
                    try:
                        self.tts.speak(sentence)
                    except Exception as e:
                        print(f"TTS exception: {e}")
                    
                    # Clear speaking flag after each sentence
                    with self.lock:
                        self.is_speaking = False
                    
                    # Small pause between sentences for natural speech rhythm
                    time.sleep(0.1)
                
            except queue.Empty:
                # Queue timeout, continue waiting
                pass
            except Exception as e:
                print(f"TTS thread exception: {e}")
                # Make sure to reset speaking flag on error
                with self.lock:
                    self.is_speaking = False
                    
            time.sleep(0.05)  # Small sleep to prevent CPU overuse
    
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
        """Run the chatbot in continuous mode
        
        Args:
            wake_word: Optional wake word to start interaction (e.g., "Hey Siri")
            exit_phrase: Phrase to exit the interaction
            full_response: If True, wait for the complete response before speaking
            activation_timeout: Seconds to remain active after wake word detection (0 for always active)
            debug_mode: If True, prints additional debugging information
        """
        language_display = "Chinese" if self.language.startswith("zh") else "English"
        
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
        
        # Keep track of activation state when using wake word
        self.is_active = not using_wake_word  # If not using wake word, always active
        active_until = 0  # Timestamp when activation expires
        running = True
        self.listen_all.start()
        self.llm_thread.start()
        self.tts_thread.start()
        while running:
            print("\n" + "="*50)
            
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
            # self.listen_all.start()
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
                print("Exiting chatbot...")
            
            # If there was an error, report it
            elif not result["success"] and result["error"]:
                print("No message")
                
                # if language_display == "Chinese":
                #     error_msg = "我遇到了一个错误。请再试一次。"
                # else:
                #     error_msg = "I encountered an error. Please try again."
                    
                # self.speak(error_msg)
            
            # Brief pause between interactions
            print("Pausing for a moment before next interaction...")
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
        # Stop camera detector if it was initialized
        if self.use_camera_emotion and self.camera_detector is not None:
            try:
                self.camera_detector.stop()
                print("Camera emotion detection stopped")
            except Exception as e:
                print(f"Error stopping camera detection: {e}")