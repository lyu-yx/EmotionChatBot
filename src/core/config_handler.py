import os
import json
from typing import Dict, Any, Optional


class ConfigHandler:
    """Configuration handler for the voice-interactive chatbot system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration handler
        
        Args:
            config_path: Path to the configuration file (default: config.json in project root)
        """
        self.config_path = config_path
        if self.config_path is None:
            # Try to locate the config in the project root
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.config_path = os.path.join(root_dir, "config.json")
        
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from the config file
        
        Returns:
            Configuration dict
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"Warning: Config file not found at {self.config_path}")
                return self._default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._default_config()
    
    def save_config(self) -> bool:
        """Save current configuration to the config file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value
        
        Args:
            section: Configuration section (e.g., 'system', 'asr', 'tts', 'llm')
            key: Key within the section
            default: Default value if not found
            
        Returns:
            The configuration value or default if not found
        """
        try:
            return self.config[section][key]
        except (KeyError, TypeError):
            return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value
        
        Args:
            section: Configuration section (e.g., 'system', 'asr', 'tts', 'llm')
            key: Key within the section
            value: Value to set
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def _default_config(self) -> Dict[str, Any]:
        """Create default configuration
        
        Returns:
            Default configuration dict
        """
        return {
            "system": {
                "debug": False,
                "exit_phrase": "exit"
            },
            "asr": {
                "engine": "alibaba",
                "language": "zh-cn",
                "timeout": 5,
                "phrase_time_limit": 15,
                "energy_threshold": 300
            },
            "tts": {
                "engine": "alibaba",
                "voice": "xiaoyun",
                "speech_rate": 0,
                "pitch_rate": 0,
                "volume": 50
            },
            "llm": {
                "provider": "alibaba",
                "model": "qwen-turbo",
                "temperature": 0.7,
                "system_prompt": "你是一个有用的语音交互助手。请提供简洁明了的回答，适合语音输出。保持回应简短但信息丰富和有帮助。"
            }
        }