import abc
from typing import Optional, Dict, Any
import os
import time
import tempfile
import json
import pyaudio
import wave
from dotenv import load_dotenv
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback, AudioFormat
import threading
from src.core.SharedQueue import SharedQueue as q
# Import the real-time MP3 player
from .realtime_player import RealtimeMp3Player
import queue

class TextToSpeech(abc.ABC):
    """Abstract base class for text-to-speech engines"""

    @abc.abstractmethod
    def speak(self, text: str) -> Dict[str, Any]:
        """Convert text to speech and speak it
        
        Args:
            text: The text to convert to speech
            
        Returns:
            Dict with at least:
                'success': Boolean indicating success status
                'error': Error message if any (None if success)
        """
        pass


class PyttsxSpeechSynthesizer(TextToSpeech):
    """Text-to-speech synthesizer using pyttsx3 engine as a fallback"""
    
    def __init__(self, voice_id=None, rate=150, volume=1.0):
        """Initialize pyttsx3 speech synthesizer
        
        Args:
            voice_id: ID of the voice to use (None for default)
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            
            # Configure voice properties
            if voice_id:
                self.engine.setProperty('voice', voice_id)
            
            # Set rate (speed)
            self.engine.setProperty('rate', rate)
            
            # Set volume (0.0 to 1.0)
            self.engine.setProperty('volume', volume)
            
            self.engine_initialized = True
            print("Pyttsx3 TTS engine initialized successfully")
        except Exception as e:
            print(f"Failed to initialize pyttsx3: {e}")
            self.engine_initialized = False
    
    def speak(self, text: str) -> Dict[str, Any]:
        """Convert text to speech and speak it using pyttsx3
        
        Args:
            text: The text to convert to speech
            
        Returns:
            Dict with:
                'success': Boolean indicating success status
                'error': Error message if any (None if success)
        """
        result = {
            "success": False,
            "error": None
        }
        
        if not self.engine_initialized:
            result["error"] = "TTS engine not properly initialized"
            print(result["error"])
            print(f"[SPOKEN TEXT]: {text}")
            return result
        
        try:
            print(f"Speaking (fallback TTS): {text}")
            self.engine.say(text)
            self.engine.runAndWait()
            result["success"] = True
        except Exception as e:
            result["error"] = f"Error during fallback speech synthesis: {e}"
            print(f"Fallback TTS error: {e}")
            print(f"[SPOKEN TEXT]: {text}")
            
        return result


class StreamingTTSSynthesizer(TextToSpeech):
    """Streaming TTS implementation using DashScope API"""
    
    def __init__(self, voice="loongstella", model="cosyvoice-v1"):
        """Initialize streaming TTS synthesizer
        
        Args:
            voice: Voice ID to use (default: "loongstella")
            model: TTS model to use (default: "cosyvoice-v1")
        """
        # Get the path to the .env file or config.json in the project root
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_path = os.path.join(root_dir, ".env")
        config_path = os.path.join(root_dir, "config.json")
        
        # Load environment variables with override=True
        load_dotenv(env_path, override=True)
        
        # Try to get API key from environment variables or config file
        self.api_key = os.getenv("ALIBABA_API_KEY")
        
        if not self.api_key and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'dashscope' in config and 'api_key' in config['dashscope']:
                        self.api_key = config['dashscope']['api_key']
                        print(f"TTS using API key from config.json: {self.api_key[:5]}...")
            except Exception as e:
                print(f"Error loading config.json: {e}")
        
        if not self.api_key:
            print("Warning: ALIBABA_API_KEY not found in environment variables or config.json")
            print("TTS will fall back to offline synthesis")
        else:
            # Set DashScope API key
            try:
                import dashscope
                dashscope.api_key = self.api_key
                print(f"Streaming TTS configured with API key: {self.api_key[:5]}...")
            except ImportError:
                print("Warning: dashscope module not found. Please install with pip install dashscope")
                self.api_key = None
        
        # Store parameters
        self.voice = voice
        self.model = model
        
        # Initialize the fallback synthesizer
        self.fallback_tts = None
        
        # Test ffmpeg availability
        self._test_ffmpeg()
        
        # Available voice list for reference
        self.available_voices = [
            "loongstella",  # Female voice (default)
            "xiaomo",       # Female voice
            "xiaochen",     # Male voice
            "xiaoxuan",     # Female voice
            "xiaoyuan",     # Female voice
            "sijia",        # Female voice
            "aiqi",         # Male voice
            "aida",         # Male voice
            "aijia",        # Female voice
            "xiaobei",      # Female voice
            "aifeng"        # Female voice
        ]
        self.interrupt = False
        self.thread_stop_event = threading.Event()
        self.interrupt_event = threading.Event()
        self.interrupt_word = "停"
        self.queue = q()
        self.is_speaking = False
    def checking_interrupt(self, synthesizer:SpeechSynthesizer ):
        while not self.thread_stop_event.is_set():
            try:
                result = self.queue.peek()
                if result:
                    if self.interrupt_word in result["text"]:
                        self.interrupt = True
                        self.interrupt_event.set()
                        if getattr(synthesizer, "_is_started", False):
                            try:
                                synthesizer.streaming_cancel()
                                self.queue.get()
                            except Exception as e:
                                print(f"Cancel failed: {e}")
            except queue.Empty:
                continue
            time.sleep(0.3)
                
    
    def _test_ffmpeg(self):
        """Test if ffmpeg is available for audio playback"""
        try:
            # Create a test player instance
            test_player = RealtimeMp3Player(verbose=False)
            has_ffmpeg = test_player.ffmpeg_path is not None
            
            if not has_ffmpeg:
                print("WARNING: ffmpeg not found. Audio playback may not work.")
                print("Please install ffmpeg and make sure it's in your PATH.")
                print("On Windows: https://ffmpeg.org/download.html")
                print("On Linux: sudo apt-get install ffmpeg")
                print("On macOS: brew install ffmpeg")
            else:
                print(f"Found ffmpeg in PATH: {test_player.ffmpeg_path}")
                
            # Clean up the test player
            test_player = None
            
            return has_ffmpeg
        except Exception as e:
            print(f"Error testing ffmpeg availability: {e}")
            return False
    
    def speak(self, text: str) -> Dict[str, Any]:
        """Convert text to speech and speak it using streaming synthesis
        
        Args:
            text: The text to convert to speech
            
        Returns:
            Dict with:
                'success': Boolean indicating success status
                'error': Error message if any (None if success)
        """
        result = {
            "success": False,
            "error": None
        }
        self.thread_stop_event.clear()
        self.interrupt_event.clear()
        if not text or text.isspace():
            result["error"] = "Empty text provided"
            return result
        
        print(f"Speaking: {text}")
        
        # Check if API key is available
        if not self.api_key:
            print("No API key available, using fallback TTS")
            return self._use_fallback_tts(text)
            
        try:
            # Initialize the player for audio playback
            player = RealtimeMp3Player(verbose=False)
            
            # Check if ffmpeg is available
            if not player.ffmpeg_path:
                print("ffmpeg not available, using fallback TTS")
                return self._use_fallback_tts(text)
            
            # Start the player
            if not player.start():
                result["error"] = "Failed to start audio player"
                print(result["error"])
                return self._use_fallback_tts(text)
            
            # Create a callback for handling TTS results
            class TTSCallback(ResultCallback):
                def __init__(self):
                    self.had_data = False
                    self.error_msg = None
                
                def on_open(self):
                    pass
                
                def on_complete(self):
                    pass
                
                def on_error(self, message: str):
                    self.error_msg = f"TTS synthesis failed: {message}"
                    print(self.error_msg)
                
                def on_close(self):
                    pass
                
                def on_event(self, message):
                    pass
                
                def on_data(self, data: bytes) -> None:
                    self.had_data = True
                    player.write(data)
            # Initialize callback
            callback = TTSCallback()
            
            # Initialize TTS synthesizer - IMPORTANT: Do not specify the format parameter to use default
            try:
                synthesizer = SpeechSynthesizer(
                    model=self.model,
                    voice=self.voice,
                    callback=callback
                )
            except Exception as e:
                print(f"Error initializing TTS synthesizer: {e}")
                return self._use_fallback_tts(text)
            self.check_interrupt = threading.Thread(
                target=self.checking_interrupt,
                args=(synthesizer,)
            )   
            self.check_interrupt.daemon = True
            self.check_interrupt.start()
            # Process text in chunks for better streaming
            max_chunk_size = 100
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            # if self.interrupt:
            #     synthesizer.streaming_cancel()
            #     self.interrupt = True
            for chunk in chunks:
                # Send text to TTS engine
                try:
                    synthesizer.streaming_call(chunk)
                    self.is_speaking = True
                except Exception as e:
                    print(f"Error in streaming_call: {e}")
                    continue
            # Signal completion
            try:
                synthesizer.streaming_complete()
            except Exception as e:
                print(f"Error in streaming_complete: {e}")
            
            # Check if synthesis was successful
            if callback.had_data:
                result["success"] = True
                # Give some time for the audio to finish playing
                time.sleep(0.5)
            elif self.interrupt == False:
                result["error"] = callback.error_msg or "No audio data produced"
                # Fall back to offline TTS if needed
                return self._use_fallback_tts(text)
            else:
                result["success"] = True
        except Exception as e:
            result["error"] = f"Error during speech synthesis: {e}"
            print(f"TTS Error: {e}")
            
            # Fall back to offline TTS in case of error
            return self._use_fallback_tts(text)
        
        finally:
            # Make sure to stop the player
            try:
                self.interrupt = False
                #self.thread_stop_event.set()
                self.is_speaking = False
                if 'player' in locals() and player is not None:
                    player.stop()
            except Exception as e:
                print(f"Error stopping player: {e}")
        
        return result
    
    def _use_fallback_tts(self, text):
        """Use offline TTS fallback
        
        Args:
            text: Text to speak
            
        Returns:
            Dict with result
        """
        result = {
            "success": False,
            "error": None
        }
        
        try:
            print("Using offline fallback for TTS...")
            if not self.fallback_tts:
                self.fallback_tts = PyttsxSpeechSynthesizer()
            
            fallback_result = self.fallback_tts.speak(text)
            result["success"] = fallback_result["success"]
            result["error"] = fallback_result.get("error")
            
        except Exception as fallback_error:
            print(f"Offline fallback also failed: {fallback_error}")
            # Print the text if all else fails
            print(f"[SPOKEN TEXT]: {text}")
            result["error"] = f"Fallback TTS failed: {fallback_error}"
            
        return result
    
    def play_audio_file(self, file_path):
        """Play an audio file
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Boolean indicating success
        """
        try:
            # Check file extension
            if file_path.lower().endswith('.wav'):
                self._play_wav_file(file_path)
                return True
            else:
                print(f"Unsupported audio format: {file_path}")
                return False
        except Exception as e:
            print(f"Error playing audio file: {e}")
            return False
    
    def _play_wav_file(self, wav_file):
        """Play a WAV file using PyAudio
        
        Args:
            wav_file: Path to the WAV file
        """
        try:
            # Open the wave file
            wf = wave.open(wav_file, 'rb')
            
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            
            # Open a stream
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            
            # Read data and play
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
                
            # Close everything
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            return True
        except Exception as e:
            print(f"Error playing WAV file: {e}")
            return False