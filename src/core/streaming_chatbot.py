"""
Enhanced Streaming Voice Chatbot
-------------------------------
An improved version of the voice chatbot that uses streaming LLM and TTS
for more natural real-time interactions. Includes better configuration,
logging, and structure.
"""

# Standard library imports
import os
import time
import logging
from typing import Optional, List, Dict, Any, Callable

# Local/Project specific imports (assuming these exist in 'src')
# from src.asr import SpeechRecognizer, AlibabaSpeechRecognizer # Example
# from src.tts import TextToSpeech, StreamingTTSSynthesizer, PyttsxSpeechSynthesizer # Example
# from src.llm import LanguageModel, StreamingLanguageModel # Example

# --- Placeholder Classes (Replace with your actual imports) ---
# These are basic stand-ins so the code runs without the actual src files.
# Replace these with your actual classes from src.*
class SpeechRecognizer:
    def recognize_from_microphone(self) -> Dict[str, Any]:
        print("Mock ASR: Listening...")
        time.sleep(2) # Simulate listening
        text = input("Mock ASR: Please type your input: ")
        print(f"Mock ASR: Recognized '{text}'")
        return {"success": True, "text": text, "error": None}
    def close(self): pass # Add close method if needed

class AlibabaSpeechRecognizer(SpeechRecognizer): # Example specific recognizer
    def __init__(self, language: str): self.language = language

class TextToSpeech:
    def speak(self, text: str):
        print(f"Mock TTS Speaking: {text}")
        # Simulate speaking time based on text length
        time.sleep(len(text) * 0.05)
    def close(self): pass # Add close method if needed

class StreamingTTSSynthesizer(TextToSpeech): # Example specific TTS
    def __init__(self, voice: str, model: str): self.voice, self.model = voice, model

class LanguageModel:
    def generate_stream_response(self, user_input: str, conversation_history: List[Dict[str, str]], chunk_callback: Callable[[str], None]) -> Dict[str, Any]:
        print(f"Mock LLM: Generating response for '{user_input}'")
        response_parts = ["Okay, ", "let me think ", "about that. ", "Hmm, ", f"you asked about '{user_input}'. ", "That's an interesting ", "topic! ", "I don't have much ", "to say right now."]
        full_response = "".join(response_parts)
        for part in response_parts:
            time.sleep(0.1) # Simulate streaming delay
            chunk_callback(part)
        print(f"Mock LLM: Full response generated.")
        return {"success": True, "response": full_response, "error": None}
    def close(self): pass # Add close method if needed

class StreamingLanguageModel(LanguageModel): # Example specific LLM
     def __init__(self, model_name: str, temperature: float, system_prompt: str):
         self.model_name = model_name
         self.temperature = temperature
         self.system_prompt = system_prompt
# --- End Placeholder Classes ---


# --- Configuration Constants ---
# Roles for conversation history
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# Default configuration values (Consider moving to a config file or env variables)
DEFAULT_LANGUAGE = os.getenv("CHATBOT_LANGUAGE", "en") # Default to English
DEFAULT_ASR_LANG_MAP = {"en": "en-US", "zh": "zh-cn"}
DEFAULT_TTS_VOICE_MAP = {"en": "en-US-Neural2-A", "zh": "loongstella"} # Example voices
DEFAULT_TTS_MODEL_MAP = {"en": "google-tts", "zh": "cosyvoice-v1"} # Example models
DEFAULT_LLM_MODEL = os.getenv("CHATBOT_LLM_MODEL", "qwen-turbo") # Example model
DEFAULT_LLM_TEMPERATURE = 0.7
DEFAULT_MAX_HISTORY_TURNS = 10 # Number of user/assistant pairs
DEFAULT_INTERACTION_PAUSE_S = 1.0 # Seconds
DEFAULT_MIN_SYNTHESIS_CHUNK_SIZE = 20 # Characters

# Default Prompts (Language specific)
DEFAULT_PROMPTS = {
    "en": {
        "system": (
            "You are a helpful voice interaction assistant. Provide clear, concise answers suitable for voice output. "
            "Keep responses relatively brief but informative and helpful."
        ),
        "greeting": "Hello! I am your voice assistant. How can I help you today?",
        "goodbye": "Goodbye! Have a great day.",
        "error": "I encountered an error. Please try again.",
    },
    "zh": {
        "system": (
            "你是一个有用的语音交互助手。请提供清晰、简洁、适合语音输出的回答。"
            "保持回应简短但信息丰富和有帮助。"
        ),
        "greeting": "你好! 我是您的语音助手。有什么可以帮您的吗？",
        "goodbye": "再见! 祝您有美好的一天。",
        "error": "我遇到了一个错误。请再试一次。",
    }
}

# Sentence terminators for streaming TTS
SENTENCE_END_CHARS = ["。", "！", "？", ".", "!", "?", "；", ";", "\n"]


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class EnhancedStreamingVoiceChatbot:
    """
    Enhanced streaming voice chatbot with real-time LLM-to-TTS capabilities,
    improved configuration, logging, and error handling.
    """

    def __init__(self,
                 recognizer: Optional[SpeechRecognizer] = None,
                 tts: Optional[TextToSpeech] = None,
                 llm: Optional[LanguageModel] = None,
                 language: str = DEFAULT_LANGUAGE,
                 system_prompt: Optional[str] = None,
                 max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
                 interaction_pause_s: float = DEFAULT_INTERACTION_PAUSE_S,
                 min_synthesis_chunk_size: int = DEFAULT_MIN_SYNTHESIS_CHUNK_SIZE
                 ):
        """Initialize the streaming voice chatbot.

        Args:
            recognizer: Speech recognition engine. If None, creates a default.
            tts: Text-to-speech engine. If None, creates a default.
            llm: Language model. If None, creates a default.
            language: The primary language for interaction ('en', 'zh', etc.).
                      Affects default prompts and potentially engine settings.
            system_prompt: Optional system prompt to guide the LLM. If None,
                           uses a default based on the language.
            max_history_turns: Max number of conversation turns (user+assistant) to keep.
            interaction_pause_s: Pause duration in seconds after each interaction.
            min_synthesis_chunk_size: Min chars in buffer to trigger TTS synthesis
                                      if no sentence end punctuation is found.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.language = language.lower()
        self.prompts = DEFAULT_PROMPTS.get(self.language, DEFAULT_PROMPTS["en"]) # Fallback to English

        # --- Default System Prompt ---
        if system_prompt is None:
            system_prompt = self.prompts["system"]
        self.logger.info(f"Using language: {self.language}")
        self.logger.info(f"System prompt: {system_prompt[:100]}...") # Log beginning

        # --- Initialize Components ---
        # Use provided components or create defaults based on language/config
        # Note: Real implementation might need API keys from env variables here
        asr_lang = DEFAULT_ASR_LANG_MAP.get(self.language, DEFAULT_ASR_LANG_MAP["en"])
        self.recognizer = recognizer if recognizer else AlibabaSpeechRecognizer(language=asr_lang)
        self.logger.info(f"Using Recognizer: {self.recognizer.__class__.__name__} with lang={asr_lang}")

        tts_voice = DEFAULT_TTS_VOICE_MAP.get(self.language, DEFAULT_TTS_VOICE_MAP["en"])
        tts_model = DEFAULT_TTS_MODEL_MAP.get(self.language, DEFAULT_TTS_MODEL_MAP["en"])
        self.tts = tts if tts else StreamingTTSSynthesizer(voice=tts_voice, model=tts_model)
        self.logger.info(f"Using TTS: {self.tts.__class__.__name__} with voice={tts_voice}, model={tts_model}")

        self.llm = llm if llm else StreamingLanguageModel(
            model_name=DEFAULT_LLM_MODEL,
            temperature=DEFAULT_LLM_TEMPERATURE,
            system_prompt=system_prompt
        )
        self.logger.info(f"Using LLM: {self.llm.__class__.__name__} with model={DEFAULT_LLM_MODEL}")

        # --- State Variables ---
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = max_history_turns * 2 # Store user+assistant msgs
        self.text_buffer: str = ""
        self.sentence_end_chars: List[str] = SENTENCE_END_CHARS
        self.min_synthesis_chunk_size = min_synthesis_chunk_size
        self.interaction_pause_s = interaction_pause_s

        self.logger.info("Enhanced Streaming Voice Chatbot initialized.")

    def listen(self) -> Dict[str, Any]:
        """Listen for user input via microphone.

        Returns:
            Dict with keys 'success' (bool), 'text' (str|None), 'error' (str|None).
        """
        self.logger.debug("Starting speech recognition...")
        try:
            result = self.recognizer.recognize_from_microphone()
            if result.get("success"):
                self.logger.info(f"Recognition successful: '{result.get('text')}'")
            else:
                self.logger.warning(f"Recognition failed: {result.get('error')}")
            return result
        except Exception as e:
            self.logger.exception("Exception during speech recognition.")
            return {"success": False, "text": None, "error": f"ASR Error: {e}"}

    def process_streaming(self, user_input: str) -> Dict[str, Any]:
        """Process user input with streaming LLM and TTS.

        Args:
            user_input: User's text input.

        Returns:
            Dict with keys 'success' (bool), 'response' (str|None), 'error' (str|None).
        """
        result = {
            "user_input": user_input,
            "response": None,
            "success": False,
            "error": None
        }
        self.text_buffer = "" # Clear buffer for new response
        full_response = []

        try:
            self.logger.info(f"Processing user input: '{user_input}'")

            # --- Define the callback for handling LLM text chunks ---
            def process_chunk(chunk: str):
                nonlocal full_response
                if not isinstance(chunk, str): # Basic type check
                     self.logger.warning(f"Received non-string chunk: {type(chunk)}. Skipping.")
                     return

                full_response.append(chunk)
                self.text_buffer += chunk
                self.logger.debug(f"Received chunk: '{chunk}', Buffer size: {len(self.text_buffer)}")

                # Check if we should synthesize and speak this part
                if self._should_synthesize():
                    text_to_synthesize = self.text_buffer
                    self.text_buffer = "" # Clear buffer *before* speaking
                    self.logger.debug(f"Synthesizing chunk: '{text_to_synthesize}'")
                    try:
                        self.tts.speak(text_to_synthesize)
                    except Exception as e:
                        self.logger.exception(f"TTS Error during streaming for chunk: '{text_to_synthesize}'")
                        # Decide if we should stop or continue? For now, log and continue.

            # --- Call the language model with streaming ---
            self.logger.debug("Calling LLM generate_stream_response...")
            llm_result = self.llm.generate_stream_response(
                user_input=user_input,
                conversation_history=self.conversation_history,
                chunk_callback=process_chunk
            )
            self.logger.debug("LLM stream finished.")

            # --- Process any remaining text in the buffer ---
            if self.text_buffer:
                self.logger.debug(f"Synthesizing remaining buffer: '{self.text_buffer}'")
                try:
                    self.tts.speak(self.text_buffer)
                except Exception as e:
                     self.logger.exception(f"TTS Error for final buffer: '{self.text_buffer}'")
                self.text_buffer = ""

            # --- Finalize Result ---
            final_response_text = "".join(full_response)
            result["response"] = final_response_text
            result["success"] = llm_result.get("success", False)
            result["error"] = llm_result.get("error") # Pass LLM error if any

            if result["success"]:
                self.logger.info(f"LLM processing successful. Full response length: {len(final_response_text)}")
                # Update conversation history
                self._update_history(user_input, final_response_text)
            else:
                self.logger.error(f"LLM processing failed: {result['error']}")

        except Exception as e:
            self.logger.exception("Exception during streaming processing.")
            result["error"] = f"Streaming Processing Error: {e}"
            result["success"] = False

        return result

    def _should_synthesize(self) -> bool:
        """Determine if the current buffer should be synthesized.

        Checks if the text buffer contains sentence-ending punctuation
        or exceeds the minimum chunk size.

        Returns:
            Boolean indicating if synthesis should occur.
        """
        if not self.text_buffer:
            return False

        # Check for sentence ending punctuation anywhere in the buffer
        if any(char in self.text_buffer for char in self.sentence_end_chars):
            self.logger.debug(f"Synthesis triggered: Sentence end found in buffer.")
            return True

        # Check if buffer is long enough
        if len(self.text_buffer) >= self.min_synthesis_chunk_size:
             self.logger.debug(f"Synthesis triggered: Buffer size >= {self.min_synthesis_chunk_size}.")
             return True

        return False

    def _update_history(self, user_input: str, assistant_response: str):
        """Adds the latest interaction to the conversation history and trims it."""
        self.conversation_history.append({"role": ROLE_USER, "content": user_input})
        self.conversation_history.append({"role": ROLE_ASSISTANT, "content": assistant_response})

        # Trim history if it exceeds maximum length
        current_length = len(self.conversation_history)
        if current_length > self.max_history_length:
            excess = current_length - self.max_history_length
            # Ensure we remove pairs (remove from the beginning)
            num_to_remove = excess if excess % 2 == 0 else excess + 1
            self.conversation_history = self.conversation_history[num_to_remove:]
            self.logger.debug(f"Trimmed conversation history. New length: {len(self.conversation_history)}")

    def run_once(self) -> Dict[str, Any]:
        """Run one complete interaction cycle: Listen -> Process -> Respond.

        Returns:
            Dict containing 'user_input', 'response', 'success', and 'error'.
        """
        result = {
            "user_input": None,
            "response": None,
            "success": False,
            "error": None
        }

        # --- Step 1: Listen ---
        listen_result = self.listen()
        if not listen_result.get("success"):
            result["error"] = listen_result.get("error", "Unknown ASR error")
            # Optional: Speak an error message if listening failed badly?
            # self.speak_error_message()
            return result # Cannot proceed without input

        user_input = listen_result.get("text")
        if not user_input: # Handle cases where ASR succeeds but returns empty text
             self.logger.warning("ASR returned empty text. Skipping processing.")
             result["error"] = "No speech detected or recognized as empty."
             # Don't treat as a full failure, just nothing to do.
             # We could optionally set success=True here if no error message needed.
             return result

        result["user_input"] = user_input

        # --- Step 2: Process and Respond ---
        process_result = self.process_streaming(user_input)

        result["response"] = process_result.get("response")
        result["success"] = process_result.get("success")
        result["error"] = process_result.get("error") # Overwrite ASR error if LLM/TTS fails

        if not result["success"]:
            self.logger.error(f"Processing failed for input '{user_input}'. Error: {result['error']}")
            # Speak a generic error message
            self.speak_error_message()
        else:
             self.logger.info("Interaction cycle completed successfully.")


        return result

    def speak_error_message(self):
        """Speaks a generic error message to the user."""
        error_msg = self.prompts.get("error", "An error occurred.")
        self.logger.info(f"Speaking error message: {error_msg}")
        try:
            self.tts.speak(error_msg)
        except Exception as e:
            self.logger.exception("Failed to speak the error message itself.")

    def run_continuous(self, wake_word: Optional[str] = None, exit_phrase: str = "exit"):
        """Run the chatbot in a continuous loop until exit phrase is detected.

        Args:
            wake_word: **Currently Conceptual**. If provided, the bot should ideally
                       wait for this word before starting the main listen-process loop.
                       Requires a dedicated wake word engine (e.g., Porcupine).
            exit_phrase: Phrase (case-insensitive) to stop the chatbot.
        """
        self.logger.info(f"Starting continuous mode. Language: {self.language}. Exit phrase: '{exit_phrase}'.")
        if wake_word:
            self.logger.warning(f"Wake word '{wake_word}' provided, but wake word detection is NOT YET IMPLEMENTED in this basic structure.")
            self.logger.warning("Chatbot will listen immediately without waiting for wake word.")
            # --- Wake Word Integration Point ---
            # Here you would initialize and run a wake word engine loop.
            # Example (pseudo-code):
            # wake_word_engine = WakeWordDetector(keyword=wake_word)
            # print(f"Listening for wake word '{wake_word}'...")
            # while not wake_word_engine.detected():
            #     time.sleep(0.1)
            # print("Wake word detected!")
            # Now proceed to the main loop.
            # This often requires running the detector in a separate thread/process
            # or using an asynchronous framework.
            # ------------------------------------

        # --- Initial Greeting ---
        greeting = self.prompts.get("greeting", "Hello!")
        self.logger.info(f"Speaking greeting: {greeting}")
        try:
            self.tts.speak(greeting)
        except Exception as e:
            self.logger.exception("Failed to speak initial greeting.")
            # Decide if we should exit or continue? Continue for now.

        running = True
        exit_phrase_lower = exit_phrase.lower()

        while running:
            self.logger.info("\n" + "="*50)

            # --- Optional: Wake Word Check (if implemented) ---
            # If wake word logic was active, you might re-enable it here
            # or wait for a specific "start command" after wake word.

            self.logger.info("Starting new interaction cycle. Listening...")
            print("\nPlease speak now...") # User-facing prompt

            # --- Run one interaction cycle ---
            result = self.run_once()

            # --- Handle result ---
            if result.get("success"):
                self.logger.info(f"Interaction Result: User='{result['user_input']}', Bot='{result['response'][:100]}...'")
                # Check for exit phrase in the *user's* input
                user_input_lower = (result.get("user_input") or "").lower()
                if exit_phrase_lower == user_input_lower or exit_phrase_lower in user_input_lower:
                    self.logger.info(f"Exit phrase '{exit_phrase}' detected in user input.")
                    goodbye = self.prompts.get("goodbye", "Goodbye.")
                    self.logger.info(f"Speaking goodbye message: {goodbye}")
                    try:
                        self.tts.speak(goodbye)
                    except Exception as e:
                        self.logger.exception("Failed to speak goodbye message.")
                    running = False
                    self.logger.info("Exiting continuous mode.")

            elif result.get("error"):
                # Error message should have already been spoken by run_once()
                self.logger.warning(f"Interaction cycle failed with error: {result['error']}")
            else:
                 # Case where ASR returned empty text, no error but no success. Just loop.
                 self.logger.info("No user input detected or input was empty. Listening again.")


            # --- Pause before next interaction (if still running) ---
            if running:
                self.logger.debug(f"Pausing for {self.interaction_pause_s} seconds...")
                time.sleep(self.interaction_pause_s)

        self.logger.info("Chatbot loop finished.")
        self.close() # Clean up resources

    def close(self):
        """Clean up resources used by the chatbot components."""
        self.logger.info("Closing chatbot resources...")
        if hasattr(self.recognizer, 'close'):
            try:
                self.recognizer.close()
                self.logger.debug("Recognizer closed.")
            except Exception as e:
                self.logger.exception("Error closing recognizer.")
        if hasattr(self.tts, 'close'):
            try:
                self.tts.close()
                self.logger.debug("TTS closed.")
            except Exception as e:
                self.logger.exception("Error closing TTS.")
        if hasattr(self.llm, 'close'):
            try:
                self.llm.close()
                self.logger.debug("LLM closed.")
            except Exception as e:
                self.logger.exception("Error closing LLM.")
        self.logger.info("Chatbot closed.")


# --- Example Usage ---
if __name__ == "__main__":
    print("Initializing Enhanced Streaming Voice Chatbot...")

    # --- Environment Variable Setup (Example - Set these in your system) ---
    # export CHATBOT_LANGUAGE="en"
    # export CHATBOT_LLM_MODEL="your_preferred_llm_model_name"
    # export ALIBABA_API_KEY="your_key" # If using Alibaba ASR/TTS
    # export OPENAI_API_KEY="your_key" # If using OpenAI LLM

    # Create and run the chatbot
    try:
        # You can customize language and other parameters here:
        chatbot = EnhancedStreamingVoiceChatbot(
            language="en", # Or "zh", or read from os.getenv
            # Pass custom components if needed:
            # recognizer=MyCustomRecognizer(),
            # tts=MyCustomTTS(),
            # llm=MyCustomLLM(),
        )
        chatbot.run_continuous(exit_phrase="goodbye computer") # Use a custom exit phrase
    except Exception as e:
        logging.exception("Failed to initialize or run the chatbot.")

    print("Chatbot execution finished.")
