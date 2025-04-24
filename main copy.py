#!/usr/bin/env python
"""
Emotion-Aware Streaming Chatbot
------------------------------
Main entry point for the EmotionChatBot application.
Combines ASR, LLM, TTS, and Emotion detection components.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Import our component classes directly
from src.core.chatbot import EmotionAwareStreamingChatbot
from src.asr.speech_recognition_engine import DashscopeSpeechRecognizer
from src.llm.language_model import StreamingLanguageModel
from src.tts.speech_synthesis import StreamingTTSSynthesizer
from src.emotion.emotion_detector import DashscopeEmotionDetector, TextBasedEmotionDetector


def init_api_key():
    """Initialize API key from environment variables or config file"""
    # Load environment variables
    load_dotenv()
    
    # Check if API key is available
    api_key = os.environ.get('ALIBABA_API_KEY')
    
    if api_key:
        print(f"API key loaded: {api_key[:5]}...")
        return api_key
    else:
        # Try to load from config.json
        try:
            import json
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'dashscope' in config and 'api_key' in config['dashscope']:
                        api_key = config['dashscope']['api_key']
                        print(f"API key loaded from config.json: {api_key[:5]}...")
                        os.environ['ALIBABA_API_KEY'] = api_key
                        return api_key
        except Exception as e:
            print(f"Error loading config.json: {e}")
    
    print("Warning: No API key found in environment variables or config.json")
    return None


def main():
    """Main function to set up and run the emotion-aware chatbot"""
    # Initialize API key
    api_key = init_api_key()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Emotion-Aware Streaming Chatbot")
    parser.add_argument("--language", "-l", default="zh-cn", choices=["zh-cn", "en-us"],
                        help="Language for the chatbot (zh-cn or en-us)")
    parser.add_argument("--voice", "-v", default=None,
                        help="Voice ID for TTS (e.g., loongstella, xiaomo, xiaochen)")
    parser.add_argument("--model", "-m", default="qwen-turbo",
                        help="LLM model to use (default: qwen-turbo)")
    
    # Emotion detection settings
    emotion_group = parser.add_argument_group('Emotion Detection')
    emotion_group.add_argument("--text-emotion", action="store_true", 
                        help="Enable text-based emotion detection (default: disabled)")
    emotion_group.add_argument("--camera-emotion", action="store_true", default = True,
                        help="Enable camera-based emotion detection (default: disabled)")
    emotion_group.add_argument("--camera-id", type=int, default=0,
                        help="Camera ID to use for emotion detection (default: 0)")
    emotion_group.add_argument("--show-camera", action="store_true", default = True,
                        help="Show camera feed window (default: hidden)")
    
    # Wake word and activation settings
    wake_group = parser.add_argument_group('Wake Word')
    wake_group.add_argument("--wake-word", type=str, default='你好助手',
                     help="Set a wake word to activate the chatbot (e.g., 'Hey Assistant')")
    wake_group.add_argument("--timeout", type=int, default=60,
                     help="Number of seconds to remain active after wake word detection (default: 60, 0 for always active)")
    wake_group.add_argument("--debug", action="store_true",
                     help="Enable debug mode to show detailed wake word detection information")
    
    # Other arguments
    parser.add_argument("--exit-phrase", default="退出",
                        help="Phrase to exit the chatbot (default: 'exit')")
    parser.add_argument("--full", action="store_true", default = True, 
                        help="Use full response mode instead of streaming speech chunks")
    
    args = parser.parse_args()
    
    # Set up components based on arguments
    language = args.language
    
    # Select appropriate voice for the language if not specified
    if args.voice is None:
        if language.startswith("zh"):
            voice = "loongstella"
        else:
            voice = "xiaomo"
    else:
        voice = args.voice
    
    # Create components
    recognizer = DashscopeSpeechRecognizer(language=language)
    
    tts = StreamingTTSSynthesizer(voice=voice, model="cosyvoice-v1")
    
    # Choose system prompt based on language
    if language.startswith("zh"):
        system_prompt = (
            "你是一个有用的、能够理解情感的语音交互助手。"
            "请根据用户的情感状态，提供适合的回答。"
            "保持回应简短但信息丰富和有帮助。"
        )
    else:
        system_prompt = (
            "You are a helpful, emotion-aware voice assistant. "
            "Provide responses appropriate to the user's emotional state. "
            "Keep responses concise but informative and helpful."
        )
    
    llm = StreamingLanguageModel(
        model_name=args.model,
        temperature=0.7,
        system_prompt=system_prompt
    )
    
    # Choose emotion detector based on preference
    if args.text_emotion:
        # Use Dashscope-based emotion detector for text
        emotion_detector = DashscopeEmotionDetector()
        print("Text-based emotion detection enabled")
    else:
        # Use simple text-based emotion detector when emotion detection is disabled
        emotion_detector = None
        print("Text-based emotion detection disabled")
    
    # Create the chatbot with appropriate camera settings
    chatbot = EmotionAwareStreamingChatbot(
        recognizer=recognizer,
        tts=tts,
        llm=llm,
        emotion_detector=emotion_detector,
        system_prompt=system_prompt,
        language=language,
        use_text_emotion=args.text_emotion,
        use_camera_emotion=args.camera_emotion,
        camera_id=args.camera_id,
        show_camera=args.show_camera
    )
    
    # Run the chatbot
    try:
        chatbot.run_continuous(
            wake_word=args.wake_word,
            exit_phrase=args.exit_phrase, 
            full_response=args.full,
            activation_timeout=args.timeout
        )
    except KeyboardInterrupt:
        print("\nExiting chatbot due to keyboard interrupt...")
    except Exception as e:
        print(f"Error running chatbot: {e}")
    finally:
        # Clean up resources
        chatbot.cleanup()
    
    print("Chatbot session ended.")


if __name__ == "__main__":
    main()