import abc
from typing import Dict, Any
import os
from dotenv import load_dotenv
import json
import time
import threading
import pyaudio
import numpy as np
from datetime import datetime
import logging
from src.core.SharedAudio import AudioManager as Audio
from src.core.SharedAudio import get_audio_manager
logging.basicConfig(level=logging.INFO)
class SpeechRecognizer(abc.ABC):
    """Abstract base class for speech recognition engines"""
    
    @abc.abstractmethod
    def recognize_from_microphone(self) -> Dict[str, Any]:
        """Recognize speech from microphone and convert to text
        
        Returns:
            Dict with at least:
                'text': The recognized text
                'success': Boolean indicating success status
                'error': Error message if any (None if success)
        """
        pass


class DashscopeSpeechRecognizer(SpeechRecognizer):
    """Speech recognition using Alibaba Dashscope ASR"""
    
    def __init__(self, language="zh-cn", timeout=10, phrase_time_limit=None):
        """Initialize Dashscope speech recognizer
        
        Args:
            language: Language code (default: "zh-cn")
            timeout: Recognition timeout in seconds
            phrase_time_limit: Maximum seconds for a phrase (None for no limit)
        """
        self.language = language
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit if phrase_time_limit else 30
        
        # Optimize recording parameters
        self.sample_rate = 16000
        self.channels = 1
        self.FORMAT = pyaudio.paInt16
        self.block_size = 800  # Reduced from 3200 for faster processing
        
        # Global variables for mic and stream
        self.mic = None
        self.stream = None
        
        # Initialize API key
        self.init_dashscope_api_key()
        
        # Initialize recorder
        self.recognition = None
        
        # Use lighter model for lower latency
        self.model = 'paraformer-realtime-v1'  # Changed from v2 to v1
        print(f"Using Dashscope ASR model: {self.model}")
        
        self.audio_manager = get_audio_manager()
        
        # Audio preprocessing buffer
        self.audio_buffer = []
        self.buffer_size = 4  # Number of blocks to buffer before processing
        
        # Optimized silence detection parameters
        self.silence_threshold = 150  # Reduced from 180 for better sensitivity
        self.silence_time_to_stop = 0.2  # Reduced from 0.3 for faster response
        self.min_speech_duration = 0.1  # Minimum duration of speech to consider
        self.max_silence_duration = 1.0  # Maximum silence duration before stopping
        self.speech_detected = False
        self.last_speech_time = None
        
        # Recording indicator control
        self.recording_active = False
        
    def init_dashscope_api_key(self):
        """Set Dashscope API key from environment variable or config file"""
        # Get the path to the .env file or config.json in the project root
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_path = os.path.join(root_dir, ".env")
        config_path = os.path.join(root_dir, "config.json")
        
        # Try to load from environment variable first
        if 'ALIBABA_API_KEY' in os.environ:
            try:
                import dashscope
                dashscope.api_key = os.environ['ALIBABA_API_KEY']
                print("Loaded Dashscope API key from environment variable")
                return
            except ImportError:
                print("Warning: dashscope module not found. Please install with pip install dashscope")
        
        # Next try config.json
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'dashscope' in config and 'api_key' in config['dashscope']:
                        import dashscope
                        dashscope.api_key = config['dashscope']['api_key']
                        print("Loaded Dashscope API key from config.json")
                        return
            except Exception as e:
                print(f"Error loading config.json: {e}")
        
        # Finally try .env file
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            if 'ALIBABA_API_KEY' in os.environ:
                try:
                    import dashscope
                    dashscope.api_key = os.environ['ALIBABA_API_KEY']
                    print("Loaded Dashscope API key from .env file")
                    return
                except ImportError:
                    print("Warning: dashscope module not found. Please install with pip install dashscope")
        
        # If we got here, no API key was found
        print("Warning: No Dashscope API key found. Please set ALIBABA_API_KEY environment variable or add to config.json")
    
    def recognize_from_microphone(self) -> Dict[str, Any]:
        """Recognize speech from microphone using Dashscope ASR
        
        Returns:
            Dict with:
                'text': The recognized text
                'success': Boolean indicating success status
                'error': Error message if any (None if success)
                'engine': 'dashscope'
        """
        result = {
            "text": "",
            "success": False, 
            "error": None,
            "engine": "dashscope"
        }
        
        try:
            import dashscope
            from dashscope.audio.asr import Recognition, RecognitionCallback
        except ImportError:
            result["error"] = "The dashscope package is not installed. Please install it with 'pip install dashscope'"
            print(result["error"])
            return result
        
        final_text = ""
        current_sentence = ""
        if not hasattr(self, "_mic"):
            self._mic = self.audio_manager
        if not hasattr(self, "_mic_stream") or self._mic_stream is None:
            self._mic_stream = self.audio_manager.get_mic_stream(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                block_size=self.block_size
            )
            
        class Callback(RecognitionCallback):
            def __init__(self):
                self.text = ""
                self.stream = None
                self.is_final = False
                self.sentence_ended = False
                self.last_text_time = time.time()
                
            def on_open(self) -> None:
                print('RecognitionCallback open.')
                
            def on_close(self) -> None:
                print('RecognitionCallback close.')
                
            def on_complete(self) -> None:
                print('RecognitionCallback completed.')
                self.is_final = True
                
            def on_error(self, message) -> None:
                print('RecognitionCallback task_id: ', message.request_id)
                print('RecognitionCallback error: ', message.message)
                result["error"] = f"Recognition error: {message.message}"
                
            def on_event(self, result_obj) -> None:
                nonlocal final_text, current_sentence
                try:
                    sentence = result_obj.get_sentence()
                    if 'text' in sentence:
                        text = sentence['text']
                        print('RecognitionCallback text: ', text)
                        current_sentence = text
                        self.last_text_time = time.time()
                        
                        # Only check for truly complete short responses (not partial ones)
                        truly_complete_responses = [
                            "没有", "有", "是的", "不是", "对的", "不对", "好的", "不好", "嗯"
                        ]
                        
                        # Check for incomplete indicators that suggest more is coming
                        incomplete_indicators = [
                            "但是", "不过", "然后", "还有", "另外", "而且", "所以", "因为", 
                            "，", "。但", "。不", "。还", "。另", "。而", "。所", "。因"
                        ]
                        
                        # Only mark as complete if it's a truly complete response AND doesn't have incomplete indicators
                        if (any(resp == text.strip() for resp in truly_complete_responses) and
                            not any(indicator in text for indicator in incomplete_indicators)):
                            if len(text.strip()) <= 5:  # Very short and complete
                                print(f"Detected truly complete short response: '{text}'")
                                self.sentence_ended = True
                        
                        if 'status_text' in sentence and sentence['status_text'] == 'SENTENCE_END':
                            print(f'RecognitionCallback sentence end: "{text}"')
                            if final_text and not final_text.endswith(("。", ".", "!", "?", "！", "？")):
                                final_text += " "
                            final_text += text
                            self.sentence_ended = True
                except Exception as e:
                    print(f"Error processing recognition result: {e}")
                
        callback = Callback()
        callback.stream = self._mic_stream
        
        try:
            recognition = Recognition(
                model=self.model,
                format='pcm',
                sample_rate=self.sample_rate,
                semantic_punctuation_enabled=True,
                callback=callback)
            
            recognition.start()
            
            print("Recording... (speak now)")
            print("This will automatically detect speech and stop after a pause.")
            print("Press Ctrl+C to stop manually.")
            
            # Start recording indicator with proper control
            self.recording_active = True
            recording_indicator_thread = threading.Thread(target=self._show_recording_indicator)
            recording_indicator_thread.daemon = True
            recording_indicator_thread.start()
            
            start_time = time.time()
            silence_start = None
            silence_duration = 0
            speech_start = None
            speech_detected = False
            consecutive_silence_blocks = 0
            max_silence_blocks = int(self.max_silence_duration * self.sample_rate / self.block_size)
            
            try:
                while time.time() - start_time < self.phrase_time_limit:
                    # Check multiple completion conditions
                    if callback.is_final or callback.sentence_ended:
                        print("\nRecognition completed, stopping early.")
                        break
                    
                    # Check if we have text but no new text for a longer while (increased time)
                    if (current_sentence and 
                        time.time() - callback.last_text_time > 2.0):  # Increased from 0.8
                        print("\nNo new text received, assuming completion.")
                        break
                    
                    # More conservative check for short complete responses
                    if current_sentence:
                        # Only stop for very specific complete responses
                        definite_complete = ["没有", "有", "嗯", "好"]
                        incomplete_signs = ["但是", "不过", "然后", "还有", "另外", "而且", "所以", "因为", "，", "。"]
                        
                        # Only stop if it's definitely complete AND no incomplete signs
                        if (any(resp == current_sentence.strip() for resp in definite_complete) and
                            not any(sign in current_sentence for sign in incomplete_signs)):
                            # Wait longer to be sure
                            if time.time() - callback.last_text_time > 1.0:  # Increased from 0.3
                                print(f"\nDetected complete short response: '{current_sentence}', stopping.")
                                break
                        
                    if hasattr(callback, 'stream') and callback.stream:
                        data = callback.stream.read(self.block_size, exception_on_overflow=False)
                        
                        # Convert audio data to numpy array for processing
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        
                        # Calculate audio energy level
                        rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
                        
                        # Speech detection logic
                        if rms >= self.silence_threshold:
                            if not speech_detected:
                                speech_start = time.time()
                                speech_detected = True
                            silence_start = None
                            silence_duration = 0
                            consecutive_silence_blocks = 0
                            self.last_speech_time = time.time()
                        else:
                            if speech_detected:
                                if silence_start is None:
                                    silence_start = time.time()
                                silence_duration = time.time() - silence_start
                                consecutive_silence_blocks += 1
                                
                                # More conservative stopping conditions
                                if current_sentence and len(current_sentence.strip()) >= 2:
                                    # Check for incomplete indicators before stopping
                                    incomplete_signs = ["但是", "不过", "然后", "还有", "另外", "而且", "所以", "因为", "，"]
                                    
                                    # If text contains incomplete indicators, wait longer
                                    if any(sign in current_sentence for sign in incomplete_signs):
                                        required_silence = 1.5  # Wait longer for incomplete sentences
                                    else:
                                        required_silence = 0.8  # Normal wait time
                                    
                                    if silence_duration > required_silence:
                                        print(f"\nSpeech ended due to silence after getting text: '{current_sentence}'")
                                        break
                                
                                # Original stopping conditions for longer responses
                                if ((silence_duration > self.silence_time_to_stop and 
                                     speech_start and (time.time() - speech_start) >= self.min_speech_duration) or
                                    consecutive_silence_blocks >= max_silence_blocks):
                                    print("\nSpeech ended due to silence detection.")
                                    break
                        
                        # Send audio frame to recognition service
                        recognition.send_audio_frame(data)
                    else:
                        time.sleep(0.05)  # Reduced sleep time for faster response
                        
            except KeyboardInterrupt:
                print("\nStopped recording due to user interrupt.")
            
            # Stop recording indicator
            self.recording_active = False
            
            recognition.stop()
            time.sleep(0.1)  # Brief wait for cleanup
            
            # Clear the recording indicator line
            print("\r" + " " * 50 + "\r", end="")
            
            recognized_text = final_text if final_text else current_sentence
            
            if recognized_text:
                result["text"] = recognized_text.strip()
                result["success"] = True
                print(f"Dashscope recognized: {recognized_text}")
            else:
                result["error"] = "No speech detected or recognized"
                print("No speech detected or recognized.")
        
        except Exception as e:
            # Make sure to stop recording indicator on error
            self.recording_active = False
            result["error"] = f"Error during speech recognition: {e}"
            print(f"\nError during Dashscope speech recognition: {e}")
        
        return result
    
    def _show_recording_indicator(self):
        """Show a simple recording indicator in the console"""
        indicators = ["🎙️ ", "🎙️  .", "🎙️  ..", "🎙️  ..."]
        i = 0
        while self.recording_active:
            print(f"\rRecording {indicators[i % len(indicators)]}", end="")
            i += 1
            time.sleep(0.25)
    
    def _get_simulated_response(self) -> Dict[str, Any]:
        """Generate a simulated response when the real ASR fails
        
        Returns:
            Dict with simulated recognition results
        """
        # Different responses based on language
        if self.language.startswith("zh"):
            # Chinese sample responses
            samples = [
                "你好，我很高兴见到你。",
                "今天天气真好。",
                "我想去公园散步。",
                "这是一个有趣的对话。",
                "我喜欢学习新的语言。",
                "请问现在几点了？",
                "我需要帮助解决这个问题。",
                "谢谢你的关心。",
                "我们应该一起吃晚饭。",
                "这个项目非常重要。"
            ]
        else:
            # English sample responses
            samples = [
                "Hello, nice to meet you.",
                "The weather is great today.",
                "I would like to go for a walk in the park.",
                "This is an interesting conversation.",
                "I enjoy learning new languages.",
                "What time is it right now?",
                "I need help solving this problem.",
                "Thank you for your concern.",
                "We should have dinner together.",
                "This project is very important."
            ]
            
        # Select a random response
        import random
        text = random.choice(samples)
        
        return {
            "text": text,
            "success": True,
            "error": None,
            "engine": "simulation"
        }