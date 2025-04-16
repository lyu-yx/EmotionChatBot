"""
Streaming Voice Chatbot
----------------------
An improved version of the voice chatbot that uses streaming LLM and TTS 
for more natural real-time interactions.
"""

from typing import Optional, List, Dict, Any
import os
import time

from src.asr import SpeechRecognizer, AlibabaSpeechRecognizer
from src.tts import TextToSpeech, StreamingTTSSynthesizer, PyttsxSpeechSynthesizer
from src.llm import LanguageModel, StreamingLanguageModel


class StreamingVoiceChatbot:
    """Streaming voice chatbot with real-time LLM-to-TTS capabilities"""
    
    def __init__(self, 
                 recognizer: Optional[SpeechRecognizer] = None,
                 tts: Optional[TextToSpeech] = None,
                 llm: Optional[LanguageModel] = None,
                 system_prompt: Optional[str] = None):
        """Initialize the streaming voice chatbot
        
        Args:
            recognizer: Speech recognition engine (default: AlibabaSpeechRecognizer)
            tts: Text-to-speech engine (default: StreamingTTSSynthesizer)
            llm: Language model (default: StreamingLanguageModel)
            system_prompt: Optional system prompt to guide the model
        """
        # Default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "你是一个有用的语音交互助手。请提供清晰、简洁、适合语音输出的回答。"
                "保持回应简短但信息丰富和有帮助。"
            )
        
        # Initialize speech recognition engine
        self.recognizer = recognizer if recognizer else AlibabaSpeechRecognizer(language="zh-cn")
        
        # Initialize streaming text-to-speech engine
        self.tts = tts if tts else StreamingTTSSynthesizer(voice="loongstella", model="cosyvoice-v1")
        
        # Initialize streaming language model
        self.llm = llm if llm else StreamingLanguageModel(
            model_name="qwen-turbo", 
            temperature=0.7,
            system_prompt=system_prompt
        )
        
        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []
        
        # Maximum number of conversation turns to keep in history
        self.max_history_length = 10
        
        # Buffer for collecting LLM output chunks
        self.text_buffer = ""
        
        # Sentence ending punctuation for splitting streaming text
        self.sentence_end_chars = ["。", "！", "？", ".", "!", "?", "；", ";"]
        
        print("Streaming Voice Chatbot initialized")
    
    def listen(self) -> Dict[str, Any]:
        """Listen for user input via microphone
        
        Returns:
            Dict with recognition results
        """
        return self.recognizer.recognize_from_microphone()
    
    def process_streaming(self, user_input: str) -> Dict[str, Any]:
        """Process user input with streaming LLM and TTS
        
        Args:
            user_input: User's text input
            
        Returns:
            Dict with response generation results
        """
        result = {
            "user_input": user_input,
            "response": "",
            "success": False,
            "error": None
        }
        
        try:
            print(f"Processing: {user_input}")
            
            # Define the callback function for handling text chunks
            def process_chunk(chunk):
                # Add to the complete response
                self.text_buffer += chunk
                
                # Check if we have a complete sentence or enough text
                if self._should_synthesize():
                    text_to_synthesize = self.text_buffer
                    self.text_buffer = ""
                    # Synthesize and play this chunk
                    self.tts.speak(text_to_synthesize)
            
            # Call the language model with streaming enabled
            llm_result = self.llm.generate_stream_response(
                user_input=user_input,
                conversation_history=self.conversation_history,
                chunk_callback=process_chunk
            )
            
            # Process any remaining text in the buffer
            if self.text_buffer:
                self.tts.speak(self.text_buffer)
                self.text_buffer = ""
            
            # Update the result
            result["response"] = llm_result["response"]
            result["success"] = llm_result["success"]
            result["error"] = llm_result["error"]
            
            # Update conversation history if successful
            if llm_result["success"]:
                # Add user message
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
    
    def run_once(self) -> Dict[str, Any]:
        """Run one complete interaction cycle
        
        Returns:
            Dict with interaction results
        """
        result = {
            "user_input": "",
            "response": "",
            "success": False,
            "error": None
        }
        
        try:
            # Step 1: Listen for user input
            listen_result = self.listen()
            
            if not listen_result["success"]:
                result["error"] = listen_result["error"]
                return result
                
            user_input = listen_result["text"]
            result["user_input"] = user_input
            
            # Step 2: Process with streaming LLM and TTS
            process_result = self.process_streaming(user_input)
            
            if not process_result["success"]:
                result["error"] = process_result["error"]
                return result
                
            result["response"] = process_result["response"]
            
            # All steps succeeded
            result["success"] = True
            
        except Exception as e:
            result["error"] = f"Error during interaction: {e}"
            
        return result
    
    def run_continuous(self, wake_word: Optional[str] = None, exit_phrase: str = "exit"):
        """Run the chatbot in continuous mode
        
        Args:
            wake_word: Optional wake word to start interaction (not implemented yet)
            exit_phrase: Phrase to exit the interaction
        """
        print(f"Streaming Voice Chatbot started. Say '{exit_phrase}' to exit.")
        print("Starting initial greeting...")
        
        # Initial greeting
        greeting = "你好! 我是您的语音助手。有什么可以帮您的吗？"
        self.tts.speak(greeting)
        
        running = True
        while running:
            print("\n" + "="*50)
            print("Listening for your command...")
            print("Please speak now...")
            
            # Run one interaction cycle
            result = self.run_once()
            
            # Print recognition result for debugging
            if result["success"]:
                print(f"\nYou said: {result['user_input']}")
                print(f"Response: {result['response']}")
            
            # Check for exit phrase
            if result["success"] and (
                result["user_input"].lower() == exit_phrase.lower() or 
                exit_phrase.lower() in result["user_input"].lower()
            ):
                goodbye = "再见! 祝您有美好的一天。"
                self.tts.speak(goodbye)
                running = False
                print("Exiting chatbot...")
            
            # If there was an error, report it
            elif not result["success"] and result["error"]:
                print(f"Error: {result['error']}")
                error_msg = "我遇到了一个错误。请再试一次。"
                self.tts.speak(error_msg)
            
            # Longer pause between interactions
            print("Pausing for 1 second before next interaction...")
            time.sleep(1)