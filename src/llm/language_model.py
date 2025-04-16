import abc
import os
from typing import List, Dict, Any, Optional, Generator, Union, Callable
from dotenv import load_dotenv
import json
import time
import dashscope

# Load environment variables
load_dotenv()


class LanguageModel(abc.ABC):
    """Abstract base class for language model integrations"""
    
    @abc.abstractmethod
    def generate_response(self, user_input: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Generate a response to user input
        
        Args:
            user_input: The user's query or statement
            conversation_history: List of previous messages in the conversation
            
        Returns:
            Dict with at least:
                'response': The generated text response
                'success': Boolean indicating success status
                'error': Error message if any (None if success)
        """
        pass


class StreamingLanguageModel(LanguageModel):
    """Language model with streaming capabilities for real-time responses"""
    
    def __init__(self, model_name="qwen-turbo", temperature=0.7, 
                 system_prompt=None, api_key=None):
        """Initialize streaming language model
        
        Args:
            model_name: Name of the model to use (default: "qwen-turbo")
            temperature: Controls randomness (0.0-1.0)
            system_prompt: Optional system message to guide the model
            api_key: Optional API key (if not provided, will load from environment)
        """
        # Get the path to the .env file in the project root
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_path = os.path.join(root_dir, ".env")
        config_path = os.path.join(root_dir, "config.json")
        
        # Load environment variables with override=True
        load_dotenv(env_path, override=True)
        
        # Try to get API key from parameter, environment variables, or config file
        self.api_key = api_key or os.getenv("ALIBABA_API_KEY")
        
        if not self.api_key and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'dashscope' in config and 'api_key' in config['dashscope']:
                        self.api_key = config['dashscope']['api_key']
            except Exception as e:
                print(f"Error loading config.json: {e}")
        
        if not self.api_key:
            raise ValueError("API key not provided and not found in environment variables or config.json")
            
        # Configure DashScope
        dashscope.api_key = self.api_key
        
        # Store model configuration
        self.model_name = model_name
        self.temperature = temperature
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are a helpful voice-interactive assistant. "
                "Provide clear, concise responses suitable for speech output. "
                "Keep responses brief but informative and helpful."
            )
        self.system_prompt = system_prompt
    
    def generate_response(self, user_input: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Generate a response using the model
        
        Args:
            user_input: The user's query or statement
            conversation_history: List of previous messages in the conversation
            
        Returns:
            Dict with:
                'response': The generated text response
                'success': Boolean indicating success status
                'error': Error message if any (None if success)
        """
        result = {
            "response": "",
            "success": False,
            "error": None
        }
        
        try:
            # Build message list
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history)
                
            # Add the current user message
            messages.append({"role": "user", "content": user_input})
            
            # Call API without streaming
            response = dashscope.Generation.call(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            
            # Extract the response text
            if response.status_code == 200:
                result["response"] = response.output.choices[0].message.content
                result["success"] = True
            else:
                result["error"] = f"API Error: {response.code} - {response.message}"
            
        except Exception as e:
            result["error"] = f"Error during language model generation: {e}"
            
        return result
    
    def generate_stream_response(self, user_input: str, 
                                conversation_history: Optional[List[Dict[str, str]]] = None, 
                                chunk_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Generate a streaming response with real-time chunks
        
        Args:
            user_input: The user's query or statement
            conversation_history: List of previous messages in the conversation
            chunk_callback: Optional callback function to process each text chunk
            
        Returns:
            Dict with:
                'response': The complete generated text response
                'success': Boolean indicating success status
                'error': Error message if any (None if success)
        """
        result = {
            "response": "",
            "success": False,
            "error": None
        }
        
        try:
            # Build message list
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history)
                
            # Add the current user message
            messages.append({"role": "user", "content": user_input})
            
            # Call API with streaming enabled
            stream_response = dashscope.Generation.call(
                model=self.model_name,
                messages=messages,
                result_format='message',
                temperature=self.temperature,
                stream=True,  # Set to False for streaming
                incremental_output=True
            )
            
            # Process streaming response
            full_response = ""
            for response in stream_response:
                if response.status_code == 200:
                    # Get text chunk
                    chunk = response.output.choices[0].message.content
                    
                    # Add to full response
                    full_response += chunk
                    
                    # Call the callback if provided
                    if chunk_callback:
                        chunk_callback(chunk)
                else:
                    result["error"] = f"API Error: {response.code} - {response.message}"
                    return result
            
            # Store the full response
            result["response"] = full_response
            result["success"] = True
            
        except Exception as e:
            result["error"] = f"Error during streaming generation: {e}"
            
        return result
