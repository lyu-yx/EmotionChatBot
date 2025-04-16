"""
Emotion-Aware Chatbot
-------------------
Integrates ASR, LLM, TTS, and Emotion detection for a complete voice interaction experience.
"""

from typing import Optional, List, Dict, Any
import os
import time
import threading

# Import our component interfaces directly from their modules
from src.asr.speech_recognition_engine import SpeechRecognizer, DashscopeSpeechRecognizer
from src.tts.speech_synthesis import TextToSpeech, StreamingTTSSynthesizer
from src.llm.language_model import LanguageModel, StreamingLanguageModel
from src.emotion.emotion_detector import EmotionDetector, DashscopeEmotionDetector, TextBasedEmotionDetector


class EmotionAwareStreamingChatbot:
    """An emotion-aware streaming chatbot that integrates ASR, LLM, TTS, and emotion detection"""
    
    def __init__(self, 
                 recognizer: Optional[SpeechRecognizer] = None,
                 tts: Optional[TextToSpeech] = None,
                 llm: Optional[LanguageModel] = None,
                 emotion_detector: Optional[EmotionDetector] = None,
                 system_prompt: Optional[str] = None,
                 language: str = "zh-cn"):
        """Initialize the emotion-aware streaming chatbot
        
        Args:
            recognizer: Speech recognition engine (default: DashscopeSpeechRecognizer)
            tts: Text-to-speech engine (default: StreamingTTSSynthesizer)
            llm: Language model (default: StreamingLanguageModel)
            emotion_detector: Emotion detection component (default: DashscopeEmotionDetector)
            system_prompt: Optional system prompt to guide the LLM
            language: Language code for ASR (default: "zh-cn")
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
        
        # Initialize emotion detector
        self.emotion_detector = emotion_detector if emotion_detector else DashscopeEmotionDetector()
        
        # Fall back to text-based emotion detector if Dashscope is not available
        if not getattr(self.emotion_detector, 'api_key', True):
            self.emotion_detector = TextBasedEmotionDetector()
            print("Warning: Using text-based emotion detector as fallback")
        
        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []
        
        # Maximum number of conversation turns to keep in history
        self.max_history_length = 10
        
        # User emotion state tracking
        self.user_emotion = "neutral"
        
        # Buffer for collecting LLM output chunks
        self.text_buffer = ""
        
        # Sentence ending punctuation for splitting streaming text
        self.sentence_end_chars = ["。", "！", "？", ".", "!", "?", "；", ";"]
        
        # Flag to indicate if TTS is currently active
        self.is_speaking = False
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        print("Emotion-Aware Streaming Chatbot initialized")
    
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
        # Detect emotion
        emotion_result = self.emotion_detector.detect_emotion_from_text(text)
        
        if emotion_result["success"]:
            # Update the user's emotional state
            self.user_emotion = emotion_result["dominant_emotion"]
            
            # Log the detected emotion
            print(f"Detected emotion: {self.user_emotion} "
                  f"(confidence: {emotion_result['emotions'].get(self.user_emotion, 0):.2f})")
        else:
            print(f"Emotion detection failed: {emotion_result.get('error', 'Unknown error')}")
        
        return self.user_emotion
    
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
            
            # Add a small delay after speaking to avoid cutting off
            time.sleep(0.3)
            
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
    
    def run_once(self, full_response: bool = False) -> Dict[str, Any]:
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
            "user_emotion": "neutral"
        }
        
        try:
            # Ensure we're not speaking before listening
            while True:
                with self.lock:
                    if not self.is_speaking:
                        break
                print("Waiting for speech to complete before listening...")
                time.sleep(0.5)
                
            # Step 1: Listen for user input
            print("Listening for user input...")
            listen_result = self.listen()
            
            if not listen_result["success"]:
                result["error"] = listen_result["error"]
                return result
                
            user_input = listen_result["text"]
            result["user_input"] = user_input
            
            # Step 2: Process emotion
            emotion = self.process_emotion(user_input)
            result["user_emotion"] = emotion
            
            # Step 3: Process with streaming LLM and TTS
            process_result = self.process_streaming(user_input, emotion, full_response)
            
            if not process_result["success"]:
                result["error"] = process_result["error"]
                return result
                
            result["response"] = process_result["response"]
            
            # All steps succeeded
            result["success"] = True
            
        except Exception as e:
            result["error"] = f"Error during interaction: {e}"
            
        return result
    
    def run_continuous(self, wake_word: Optional[str] = None, exit_phrase: str = "exit", full_response: bool = False):
        """Run the chatbot in continuous mode
        
        Args:
            wake_word: Optional wake word to start interaction (not implemented yet)
            exit_phrase: Phrase to exit the interaction
            full_response: If True, wait for the complete response before speaking
        """
        language_display = "Chinese" if self.recognizer.language.startswith("zh") else "English"
        print(f"Emotion-Aware Chatbot started ({language_display} mode). Say '{exit_phrase}' to exit.")
        speech_mode = "full response mode" if full_response else "streaming mode"
        print(f"Speech output using {speech_mode}")
        print("Starting initial greeting...")
        
        # Initial greeting
        if language_display == "Chinese":
            greeting = "你好! 我是能够感知情绪的语音助手。请问今天我能帮您什么？"
        else:
            greeting = "Hello! I'm an emotion-aware voice assistant. How can I help you today?"
        
        self.speak(greeting)
        
        running = True
        while running:
            print("\n" + "="*50)
            print("Listening for your command...")
            print("Please speak now...")
            
            # Run one interaction cycle
            result = self.run_once(full_response)
            
            # Print recognition result for debugging
            if result["success"]:
                print(f"\nYou said: {result['user_input']} (Emotion: {result['user_emotion']})")
                print(f"Response: {result['response']}")
            
            # Check for exit phrase
            if result["success"] and (
                result["user_input"].lower() == exit_phrase.lower() or 
                exit_phrase.lower() in result["user_input"].lower()
            ):
                if language_display == "Chinese":
                    goodbye = "再见! 祝您有美好的一天。"
                else:
                    goodbye = "Goodbye! Have a great day."
                self.speak(goodbye)
                running = False
                print("Exiting chatbot...")
            
            # If there was an error, report it
            elif not result["success"] and result["error"]:
                print(f"Error: {result['error']}")
                
                if language_display == "Chinese":
                    error_msg = "我遇到了一个错误。请再试一次。"
                else:
                    error_msg = "I encountered an error. Please try again."
                    
                self.speak(error_msg)
            
            # Longer pause between interactions
            print("Pausing for 1 second before next interaction...")
            time.sleep(1)