#!/usr/bin/env python
"""
Simple LLM to TTS Streaming Integration
--------------------------------------
Simplified version matching the official Alibaba Cloud example
"""

import os
import sys
from http import HTTPStatus
from dotenv import load_dotenv

import dashscope
from dashscope import Generation
from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback

# Import our realtime player
from src.tts.realtime_player import RealtimeMp3Player

# Configuration
system_prompt = '你是一个有用的语音交互助手。请提供清晰、简洁、适合语音输出的中文回答。保持回答简短但信息丰富且有帮助。'


def init_dashscope_api_key():
    """Initialize the DashScope API key from environment variables"""
    # Load environment variables
    load_dotenv()
    
    if 'ALIBABA_API_KEY' in os.environ:
        dashscope.api_key = os.environ['ALIBABA_API_KEY']
        print(f"Loaded API key: {dashscope.api_key[:5]}...")
    else:
        dashscope.api_key = None
        print("Error: ALIBABA_API_KEY not found in environment variables")


def synthesize_speech_from_llm_by_streaming_mode(query_text: str, voice='loongstella', model='cosyvoice-v1'):
    """Synthesize speech with LLM streaming output text
    
    Args:
        query_text: Query to send to LLM
        voice: Voice ID for TTS (default: "loongstella", other options: "xiaomo", "xiaochen")
        model: TTS model to use (default: "cosyvoice-v1")
    """
    # Initialize the player
    player = RealtimeMp3Player(verbose=True)
    if not player.start():
        print("Error initializing audio player. Make sure ffmpeg is installed.")
        return

    # Create a callback handler for TTS results
    class Callback(ResultCallback):
        def __init__(self):
            self.had_data = False
            
        def on_open(self):
            pass

        def on_complete(self):
            pass

        def on_error(self, message: str):
            print(f'Speech synthesis task failed: {message}')
            if "ModelNotFound" in message:
                print("Try using a different model or voice. Available voices include: loongstella, xiaomo, xiaochen")

        def on_close(self):
            pass

        def on_event(self, message):
            pass

        def on_data(self, data: bytes) -> None:
            # Send audio data to player
            self.had_data = True
            player.write(data)

    # Initialize TTS synthesizer with the callback
    synthesizer_callback = Callback()
    
    try:
        synthesizer = SpeechSynthesizer(
            model=model,
            voice=voice,
            callback=synthesizer_callback,
        )
    except Exception as e:
        print(f"Error initializing speech synthesizer: {e}")
        player.stop()
        return

    # Prepare messages for LLM call
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': query_text}
    ]
    
    print(f">>> query: {query_text}")
    print(">>> answer: ", end='', flush=True)
    
    # Call LLM with streaming enabled
    responses = Generation.call(
        model='qwen-turbo', 
        messages=messages,
        result_format='message',
        stream=True, 
        incremental_output=True
    )
    
    # Process streaming responses
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            # Get text chunk from LLM
            llm_text_chunk = response.output.choices[0].message.content
            print(llm_text_chunk, end='', flush=True)
            
            # Send text chunk to TTS synthesizer
            synthesizer.streaming_call(llm_text_chunk)
        else:
            print(
                f"\nRequest id: {response.request_id}, "
                f"Status code: {response.status_code}, "
                f"Error code: {response.code}, "
                f"Error message: {response.message}"
            )
    
    # Complete the streaming synthesis
    synthesizer.streaming_complete()
    
    # Add a newline after LLM output
    print("")
    print(">>> playback completed")
    
    # Display performance metrics
    print(f"[Metrics] RequestID: {synthesizer.get_last_request_id()}, "
          f"First package delay: {synthesizer.get_first_package_delay()} ms")
    
    # Stop the player
    player.stop()


def main():
    """Main interactive function"""
    print("=== Simple LLM-TTS Streaming Demo ===\n")
    
    # Initialize API key
    init_dashscope_api_key()
    
    if not dashscope.api_key:
        return
    
    # Check if a query was provided as command line argument
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
        synthesize_speech_from_llm_by_streaming_mode(query)
        return
    
    # Interactive loop
    print("Type your questions (or 'exit' to quit):\n")
    
    while True:
        try:
            # Get user input
            query = input("\nYou: ").strip()
            
            # Exit if requested
            if query.lower() in ['exit', 'quit']:
                break
                
            # Skip empty input
            if not query:
                continue
                
            # Process the query
            synthesize_speech_from_llm_by_streaming_mode(query)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nDemo completed.")


if __name__ == "__main__":
    main()