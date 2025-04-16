#!/usr/bin/env python
"""
Streaming Chatbot Demo with Dashscope ASR
----------------------------------------
Demo script showing how to use the streaming chatbot with Dashscope ASR.
"""

import os
import sys
import dotenv
import argparse

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core import StreamingChatbot, ConfigHandler
from src.asr import DashscopeSpeechRecognizer
from src.tts import AlibabaSpeechSynthesizer, PyttsxSpeechSynthesizer
from src.llm import AlibabaLanguageModel, OpenAILanguageModel

def main():
    """Entry point for the streaming chatbot demo"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Streaming Chatbot Demo")
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json file")
    parser.add_argument("--exit-phrase", type=str, help="Phrase to exit the chatbot")
    args = parser.parse_args()
    
    # Load environment variables
    env_path = os.path.abspath(args.env)
    print(f"Loading environment variables from: {env_path}")
    dotenv.load_dotenv(env_path, override=True)
    
    # Verify API keys
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not found. Please set it in your .env file.")
        print("Example: DASHSCOPE_API_KEY=your_api_key_here")
        return
    else:
        print(f"Using Dashscope API key: {api_key[:5]}...")
    
    # Load configuration
    print("Loading configuration...")
    config = ConfigHandler(config_path=args.config)
    
    # Get exit phrase
    exit_phrase = args.exit_phrase if args.exit_phrase else config.get("system", "exit_phrase")
    
    # Initialize components
    print("Initializing components...")
    
    # Create speech recognizer
    recognizer = DashscopeSpeechRecognizer(
        language=config.get("asr", "language", "zh-cn"),
        timeout=config.get("asr", "timeout", 10),
        phrase_time_limit=config.get("asr", "phrase_time_limit", 30)
    )
    
    # Create speech synthesizer
    tts_engine = config.get("tts", "engine", "alibaba").lower()
    if tts_engine == "pyttsx3":
        synthesizer = PyttsxSpeechSynthesizer(
            voice_id=config.get("tts", "voice_id"),
            rate=config.get("tts", "rate", 200),
            volume=config.get("tts", "volume", 1.0)
        )
    else:  # default to alibaba
        synthesizer = AlibabaSpeechSynthesizer(
            voice=config.get("tts", "voice", "zhitian_emo"),
            speech_rate=config.get("tts", "speech_rate", 0),
            pitch_rate=config.get("tts", "pitch_rate", 0),
            volume=config.get("tts", "volume", 50)
        )
    
    # Create language model
    llm_provider = config.get("llm", "provider", "alibaba").lower()
    if llm_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not found. Please set it in your .env file.")
            print("Example: OPENAI_API_KEY=your_api_key_here")
            return
        
        language_model = OpenAILanguageModel(
            model_name=config.get("llm", "model", "gpt-3.5-turbo"),
            temperature=config.get("llm", "temperature", 0.7),
            system_prompt=config.get("llm", "system_prompt")
        )
    else:  # default to alibaba
        language_model = AlibabaLanguageModel(
            model_name=config.get("llm", "model", "qwen-turbo"),
            temperature=config.get("llm", "temperature", 0.7),
            system_prompt=config.get("llm", "system_prompt")
        )
    
    # Create streaming chatbot
    chatbot = StreamingChatbot(
        recognizer=recognizer,
        synthesizer=synthesizer,
        language_model=language_model,
        stream_output=True
    )
    
    print(f"Streaming Chatbot ready! Say '{exit_phrase}' to exit.")
    
    try:
        # Run the streaming chatbot
        chatbot.run_interactive(exit_phrase=exit_phrase)
    except KeyboardInterrupt:
        print("\nExiting Streaming Chatbot...")
    except Exception as e:
        print(f"\nError: {e}")
    
    print("Streaming Chatbot shut down.")

if __name__ == "__main__":
    main()