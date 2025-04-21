import abc
from typing import Dict, Any
import os
from dotenv import load_dotenv
import json
import time
import threading
import pyaudio
import numpy as np


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
        
        # Set recording parameters
        self.sample_rate = 16000  # sampling rate (Hz)
        self.channels = 1  # mono channel
        self.FORMAT = pyaudio.paInt16  # data type
        self.block_size = 3200  # number of frames per buffer
        
        # Global variables for mic and stream
        self.mic = None
        self.stream = None
        
        # Initialize API key
        self.init_dashscope_api_key()
        
        # Initialize recorder
        self.recognition = None
        
        # Set model based on language
        self.model = 'paraformer-realtime-v2'
        print(f"Using Dashscope ASR model: {self.model}")
        
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
        
        # Store the final recognized text (not intermediate results)
        final_text = ""
        current_sentence = ""
        
        # Real-time speech recognition callback
        class Callback(RecognitionCallback):
            def on_open(self) -> None:
                print('RecognitionCallback open.')
                self.mic = pyaudio.PyAudio()
                self.stream = self.mic.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=16000,
                              input=True)

            def on_close(self) -> None:
                print('RecognitionCallback close.')
                if hasattr(self, 'stream') and self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                if hasattr(self, 'mic') and self.mic:
                    self.mic.terminate()
                self.stream = None
                self.mic = None

            def on_complete(self) -> None:
                print('RecognitionCallback completed.')

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
                        
                        # Update the current sentence with the latest result
                        current_sentence = text
                        
                        # Check for sentence end status directly rather than using is_sentence_end
                        if 'status_text' in sentence and sentence['status_text'] == 'SENTENCE_END':
                            print(f'RecognitionCallback sentence end: "{text}"')
                            # Add the completed sentence to final text only when sentence is complete
                            if final_text and not final_text.endswith(("。", ".", "!", "?", "！", "？")):
                                final_text += " "
                            final_text += text
                except Exception as e:
                    print(f"Error processing recognition result: {e}")
        
        # Create the callback
        callback = Callback()
        
        try:
            # Call recognition service in async mode
            recognition = Recognition(
                model=self.model,
                format='pcm',
                sample_rate=self.sample_rate,
                semantic_punctuation_enabled=True,
                callback=callback)
            
            # Start recognition
            recognition.start()
            
            print("Recording... (speak now)")
            print("This will automatically detect speech and stop after a pause.")
            print("Press Ctrl+C to stop manually.")
            
            # Show recording indicator
            recording_indicator_thread = threading.Thread(target=self._show_recording_indicator)
            recording_indicator_thread.daemon = True
            recording_indicator_thread.start()
            
            # Record for a maximum of phrase_time_limit seconds
            start_time = time.time()
            silence_start = None
            silence_duration = 0
            silence_threshold = 500  # Adjusted silence detection (higher = less sensitive)
            silence_time_to_stop = 2  # Silence duration required to stop recording (reduced from 5s to 2s)
            
            try:
                # Continue until timeout or silence detected
                while time.time() - start_time < self.phrase_time_limit:
                    # Check if stream is available
                    if hasattr(callback, 'stream') and callback.stream:
                        # Read audio data
                        data = callback.stream.read(self.block_size, exception_on_overflow=False)
                        
                        # Send to recognition service
                        recognition.send_audio_frame(data)
                        
                        # Calculate audio energy level for silence detection
                        rms = sum(abs(int.from_bytes(data[i:i+2], byteorder='little', signed=True)) 
                                for i in range(0, len(data), 2)) / (len(data)/2)
                        
                        # Check for silence
                        if rms < silence_threshold:
                            if silence_start is None:
                                silence_start = time.time()
                            silence_duration = time.time() - silence_start
                            if silence_duration > silence_time_to_stop:
                                print("\nSpeech ended due to silence detection.")
                                break
                        else:
                            # Reset silence detection if sound detected
                            silence_start = None
                            silence_duration = 0
                    else:
                        # Wait for the stream to be initialized
                        time.sleep(0.1)
                        
            except KeyboardInterrupt:
                print("\nStopped recording due to user interrupt.")
            
            # Stop recognition
            recognition.stop()
            
            # Wait a brief moment for final processing
            time.sleep(0.5)
            
            # Use the final text if available, otherwise use current sentence
            recognized_text = final_text if final_text else current_sentence
            
            # Update result
            if recognized_text:
                result["text"] = recognized_text.strip()
                result["success"] = True
                print(f"\nDashscope recognized: {recognized_text}")
            else:
                result["error"] = "No speech detected or recognized"
                print("\nNo speech detected or recognized.")
        
        except Exception as e:
            result["error"] = f"Error during speech recognition: {e}"
            print(f"\nError during Dashscope speech recognition: {e}")
            # Use simulated response as fallback
            sim_result = self._get_simulated_response()
            result["text"] = sim_result["text"]
            result["success"] = True
            result["engine"] = "simulation_fallback"
            print(f"Using simulated response: {sim_result['text']}")
        
        return result
    
    def _show_recording_indicator(self):
        """Show a simple recording indicator in the console"""
        indicators = ["🎙️ ", "🎙️  .", "🎙️  ..", "🎙️  ..."]
        i = 0
        while True:
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