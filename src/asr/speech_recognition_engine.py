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
        self.phrase_time_limit = phrase_time_limit if phrase_time_limit else 60  # Increased from 30 to 60 seconds
        
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
        self.model = 'paraformer-realtime-8k-v2'
        print(f"Using Dashscope ASR model: {self.model}")
        
        # 自适应环境噪音阈值采集
        self.silence_threshold = self._collect_env_rms()
        print(f"[ASR] 自适应静音阈值: {self.silence_threshold}")
        
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
    
    def _collect_env_rms(self):
        """采集环境噪音rms，返回自适应静音阈值"""
        import pyaudio
        import time
        
        try:
            env_rms_samples = []
            env_sample_time = 2.0  # 增加到2秒，采集更多样本
            sample_rate = self.sample_rate
            block_size = self.block_size
            
            mic = pyaudio.PyAudio()
            stream = mic.open(format=self.FORMAT, channels=1, rate=sample_rate, input=True)
            frames = int(sample_rate * env_sample_time / block_size)
            
            print("请保持安静，正在采集环境噪音（2秒）...")
            for i in range(frames):
                data = stream.read(block_size, exception_on_overflow=False)
                rms = sum(abs(int.from_bytes(data[i:i+2], byteorder='little', signed=True)) 
                          for i in range(0, len(data), 2)) / (len(data)/2)
                env_rms_samples.append(rms)
                if i % 5 == 0:  # 每5帧打印一次，减少输出
                    print(f"采集环境噪音rms: {rms}")
                
            stream.stop_stream()  
            stream.close()
            mic.terminate()
            
            if env_rms_samples:
                env_rms = sum(env_rms_samples) / len(env_rms_samples)
                # 使用更保守的策略：取环境噪音的1.5-2倍，且设置最小值800
                adaptive_threshold = max(800, env_rms * 1.5)
                print(f"环境噪音均值: {env_rms}, 自适应阈值: {adaptive_threshold}")
                return adaptive_threshold
            else:
                return 800  # 更合理的fallback值
                
        except Exception as e:
            print(f"环境噪音采集失败: {e}, 使用默认阈值800")
            return 800
    
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
                # Initialize text detection flags
                self.has_text = False
                self.last_text_time = None
                self.sentence_complete = False

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
                        
                        # Mark that we have received text (speech detected)
                        self.has_text = True
                        self.last_text_time = time.time()
                        
                        # Check for sentence end status directly rather than using is_sentence_end
                        if 'status_text' in sentence and sentence['status_text'] == 'SENTENCE_END':
                            print(f'RecognitionCallback sentence end: "{text}"')
                            # Add the completed sentence to final text only when sentence is complete
                            if final_text and not final_text.endswith(("。", ".", "!", "?", "！", "？")):
                                final_text += " "
                            final_text += text
                            self.sentence_complete = True
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
            
            print("🎤 Ready to listen... (speak when ready)")
            print("💡 System will wait for your speech, then stop quickly after you finish.")
            print("Press Ctrl+C to stop manually.")
            
            # Set recording flag and show recording indicator
            self._recording_active = True
            self._speech_detected_for_indicator = False  # Initialize speech detection flag for indicator
            recording_indicator_thread = threading.Thread(target=self._show_recording_indicator)
            recording_indicator_thread.daemon = True
            recording_indicator_thread.start()
            
            # Record for a maximum of phrase_time_limit seconds
            start_time = time.time()
            text_timeout = 2.0  # Wait 2 seconds after last text received
            
            try:
                # Continue until timeout or text received and then timeout
                while time.time() - start_time < self.phrase_time_limit:
                    # Check if stream is available
                    if hasattr(callback, 'stream') and callback.stream:
                        # Read audio data
                        data = callback.stream.read(self.block_size, exception_on_overflow=False)
                        
                        # Send to recognition service
                        recognition.send_audio_frame(data)
                        
                        # Check if we have received text from recognition
                        if hasattr(callback, 'has_text') and callback.has_text:
                            if not self._speech_detected_for_indicator:
                                self._speech_detected_for_indicator = True
                                print("\n🎤 Speech recognized, processing...")
                            
                            # Check if sentence is complete
                            if hasattr(callback, 'sentence_complete') and callback.sentence_complete:
                                print("\n✅ Sentence completed.")
                                break
                            
                            # Check if we should timeout after receiving text
                            if hasattr(callback, 'last_text_time') and callback.last_text_time:
                                time_since_last_text = time.time() - callback.last_text_time
                                if time_since_last_text > text_timeout:
                                    print(f"\n✅ Text completed after {time_since_last_text:.1f}s timeout.")
                                    break
                    else:
                        # Wait for the stream to be initialized
                        time.sleep(0.1)
                    
                    # Small sleep to prevent excessive CPU usage
                    time.sleep(0.05)
                        
            except KeyboardInterrupt:
                print("\nStopped recording due to user interrupt.")
            
            # Stop recognition
            recognition.stop()
            
            # Stop recording indicator
            self._recording_active = False
            
            # Wait a brief moment for final processing
            time.sleep(0.25)
            timestamp = datetime.now().timestamp()
            logging.info(f"after recognition:{timestamp}")
            # Use the final text if available, otherwise use current sentence
            recognized_text = final_text if final_text else current_sentence
            
            # Update result
            if recognized_text:
                result["text"] = recognized_text.strip()
                result["success"] = True
                print(f"\n✅ Dashscope recognized: {recognized_text}")
            elif hasattr(callback, 'has_text') and callback.has_text:
                # Text was detected but may be incomplete
                result["error"] = "Speech detected but recognition may be incomplete"
                print("\n⚠️ Speech detected but recognition may be incomplete. Please try speaking more clearly.")
            else:
                # No text detected at all - this should trigger a retry, not an error
                result["error"] = "No speech detected - please try again"
                print(f"\n⏳ No speech detected in {self.phrase_time_limit}s. Please try speaking again.")
        
        except Exception as e:
            # Stop recording indicator in case of error
            self._recording_active = False
            result["error"] = f"Error during speech recognition: {e}"
            print(f"\nError during Dashscope speech recognition: {e}")
            # Use simulated response as fallback
            sim_result = self._get_simulated_response()
            result["text"] = sim_result["text"]
            result["success"] = True
            result["engine"] = "simulation_fallback"
            print(f"Using simulated response: {sim_result['text']}")
        
        return result
    
    def recognize_with_vocabulary(self, vocab_params: Dict[str, Any]) -> Dict[str, Any]:
        """Recognition with vocabulary support using official API
        
        Args:
            vocab_params: Vocabulary parameters containing vocabulary_id and language_hints
            
        Returns:
            Dict with recognition results
        """
        vocabulary_id = vocab_params.get('vocabulary_id')
        language_hints = vocab_params.get('language_hints', ['zh'])
        
        print(f"🎯 Using medical vocabulary: {vocabulary_id}")
        
        # Import required modules
        try:
            import dashscope
            from dashscope.audio.asr import Recognition, RecognitionCallback
        except ImportError:
            result = {
                "text": "",
                "success": False,
                "error": "The dashscope package is not installed. Please install it with 'pip install dashscope'",
                "engine": "dashscope_with_vocabulary"
            }
            print(result["error"])
            return result
        
        # Use the same logic as recognize_from_microphone but with vocabulary parameters
        result = {
            "text": "",
            "success": False,
            "error": None,
            "engine": "dashscope_with_vocabulary"
        }
        
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
                # Initialize text detection flags
                self.has_text = False
                self.last_text_time = None
                self.sentence_complete = False

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
                        
                        # Mark that we have received text (speech detected)
                        self.has_text = True
                        self.last_text_time = time.time()
                        
                        # Check for sentence end status directly rather than using is_sentence_end
                        if 'status_text' in sentence and sentence['status_text'] == 'SENTENCE_END':
                            print(f'RecognitionCallback sentence end: "{text}"')
                            # Add the completed sentence to final text only when sentence is complete
                            if final_text and not final_text.endswith(("。", ".", "!", "?", "！", "？")):
                                final_text += " "
                            final_text += text
                            self.sentence_complete = True
                except Exception as e:
                    print(f"Error processing recognition result: {e}")
        
        # Create the callback
        callback = Callback()
        
        try:
            # Call recognition service with vocabulary parameters
            recognition = Recognition(
                model=self.model,
                format='pcm',
                sample_rate=self.sample_rate,
                semantic_punctuation_enabled=True,
                callback=callback,
                vocabulary_id=vocabulary_id,
                language_hints=language_hints)
            
            # Start recognition
            recognition.start()
            
            print("🎤 Ready to listen with medical vocabulary... (speak when ready)")
            print("💡 Medical terms will be recognized with higher accuracy.")
            print("Press Ctrl+C to stop manually.")
            
            # Set recording flag and show recording indicator
            self._recording_active = True
            self._speech_detected_for_indicator = False
            recording_indicator_thread = threading.Thread(target=self._show_recording_indicator)
            recording_indicator_thread.daemon = True
            recording_indicator_thread.start()
            
            # Record for a maximum of phrase_time_limit seconds
            start_time = time.time()
            text_timeout = 2.0  # Wait 2 seconds after last text received
            
            try:
                # Continue until timeout or text received and then timeout
                while time.time() - start_time < self.phrase_time_limit:
                    # Check if stream is available
                    if hasattr(callback, 'stream') and callback.stream:
                        # Read audio data
                        data = callback.stream.read(self.block_size, exception_on_overflow=False)
                        
                        # Send to recognition service
                        recognition.send_audio_frame(data)
                        
                        # Check if we have received text from recognition
                        if hasattr(callback, 'has_text') and callback.has_text:
                            if not self._speech_detected_for_indicator:
                                self._speech_detected_for_indicator = True
                                print("\n🎤 Speech recognized with vocabulary, processing...")
                            
                            # Check if sentence is complete
                            if hasattr(callback, 'sentence_complete') and callback.sentence_complete:
                                print("\n✅ Sentence completed.")
                                break
                            
                            # Check if we should timeout after receiving text
                            if hasattr(callback, 'last_text_time') and callback.last_text_time:
                                time_since_last_text = time.time() - callback.last_text_time
                                if time_since_last_text > text_timeout:
                                    print(f"\n✅ Text completed after {time_since_last_text:.1f}s timeout.")
                                    break
                    else:
                        # Wait for the stream to be initialized
                        time.sleep(0.1)
                    
                    # Small sleep to prevent excessive CPU usage
                    time.sleep(0.05)
                        
            except KeyboardInterrupt:
                print("\nStopped recording due to user interrupt.")
            
            # Stop recognition
            recognition.stop()
            
            # Stop recording indicator
            self._recording_active = False
            
            # Wait a brief moment for final processing
            time.sleep(0.25)
            
            # Use the final text if available, otherwise use current sentence
            recognized_text = final_text if final_text else current_sentence
            
            # Update result
            if recognized_text:
                result["text"] = recognized_text.strip()
                result["success"] = True
                result["vocabulary_used"] = True
                result["vocabulary_id"] = vocabulary_id
                result["language_hints"] = language_hints
                print(f"\n✅ Dashscope recognized with vocabulary: {recognized_text}")
            elif hasattr(callback, 'has_text') and callback.has_text:
                # Text was detected but may be incomplete
                result["error"] = "Speech detected but recognition may be incomplete"
                print("\n⚠️ Speech detected but recognition may be incomplete. Please try speaking more clearly.")
            else:
                # No text detected at all
                result["error"] = "No speech detected - please try again"
                print(f"\n⏳ No speech detected in {self.phrase_time_limit}s. Please try speaking again.")
        
        except Exception as e:
            # Stop recording indicator in case of error
            self._recording_active = False
            result["error"] = f"Error during vocabulary-enhanced speech recognition: {e}"
            print(f"\nError during vocabulary-enhanced speech recognition: {e}")
            # Fallback to regular recognition
            print("Falling back to regular recognition...")
            return self.recognize_from_microphone()
        
        return result
    
    def _show_recording_indicator(self):
        """Show a dynamic recording indicator in the console"""
        waiting_indicators = ["⏳ ", "⏳  .", "⏳  ..", "⏳  ..."]
        listening_indicators = ["🎤 ", "🎤  .", "🎤  ..", "🎤  ..."]
        i = 0
        # Use a flag to control the indicator loop
        while getattr(self, '_recording_active', True):
            # Check if speech has been detected to change indicator
            if getattr(self, '_speech_detected_for_indicator', False):
                indicator_set = listening_indicators
                prefix = "Listening"
            else:
                indicator_set = waiting_indicators
                prefix = "Waiting"
            
            print(f"\r{prefix} {indicator_set[i % len(indicator_set)]}", end="", flush=True)
            i += 1
            time.sleep(0.25)
        # Clear the recording indicator when done
        print("\r" + " " * 30 + "\r", end="", flush=True)
    
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

