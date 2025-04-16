import abc
from typing import Dict, Any, Optional, List


class EmotionDetector(abc.ABC):
    """Abstract base class for emotion detection components
    
    This will serve as the interface for future emotion detection implementations.
    Future implementations might include audio-based emotion detection,
    text-based sentiment analysis, or facial expression detection.
    """
    
    @abc.abstractmethod
    def detect_emotion_from_audio(self, audio_data: Any) -> Dict[str, Any]:
        """Detect emotions from audio input
        
        Args:
            audio_data: Audio data to analyze
            
        Returns:
            Dict with at least:
                'emotions': Dict mapping emotion names to confidence scores
                'dominant_emotion': Name of the dominant emotion
                'success': Boolean indicating success status
                'error': Error message if any (None if success)
        """
        pass
    
    @abc.abstractmethod
    def detect_emotion_from_text(self, text: str) -> Dict[str, Any]:
        """Detect emotions from text input
        
        Args:
            text: Text to analyze for emotional content
            
        Returns:
            Dict with at least:
                'emotions': Dict mapping emotion names to confidence scores
                'dominant_emotion': Name of the dominant emotion
                'success': Boolean indicating success status
                'error': Error message if any (None if success)
        """
        pass
    
    @abc.abstractmethod
    def get_available_emotions(self) -> List[str]:
        """Get the list of emotions that this detector can recognize
        
        Returns:
            List of emotion names that this detector can identify
        """
        pass


class TextBasedEmotionDetector(EmotionDetector):
    """Basic text-based emotion detection using keyword patterns
    
    This is a simple implementation that detects emotions based on
    keyword patterns in text. More sophisticated implementations could
    use machine learning models for sentiment analysis.
    """
    
    def __init__(self):
        """Initialize the text-based emotion detector with keyword patterns"""
        # Define emotion keywords (simplified for demonstration)
        self.emotion_keywords = {
            "happy": ["happy", "joy", "delighted", "glad", "pleasant", "excited", "smile", "enjoy", "fun", "love"],
            "sad": ["sad", "unhappy", "unfortunate", "depressed", "gloomy", "miserable", "sorry", "regret"],
            "angry": ["angry", "mad", "furious", "irritated", "annoyed", "rage", "hate", "frustrated"],
            "afraid": ["afraid", "scared", "frightened", "panic", "terror", "fear", "worry", "anxious"],
            "surprised": ["surprised", "amazed", "astonished", "shocked", "unexpected", "wow"],
            "neutral": ["neutral", "ok", "fine", "normal", "average", "moderate"]
        }
        
        # Default emotion when nothing is detected
        self.default_emotion = "neutral"
        
    def detect_emotion_from_audio(self, audio_data: Any) -> Dict[str, Any]:
        """Detect emotions from audio input (not implemented)
        
        Args:
            audio_data: Audio data to analyze
            
        Returns:
            Dict with:
                'emotions': Empty dict
                'dominant_emotion': Default emotion
                'success': False
                'error': Error message
        """
        return {
            "emotions": {},
            "dominant_emotion": self.default_emotion,
            "success": False,
            "error": "Audio-based emotion detection not implemented"
        }
    
    def detect_emotion_from_text(self, text: str) -> Dict[str, Any]:
        """Detect emotions from text input using keyword patterns
        
        Args:
            text: Text to analyze for emotional content
            
        Returns:
            Dict with:
                'emotions': Dict mapping emotion names to confidence scores
                'dominant_emotion': Name of the dominant emotion
                'success': Boolean indicating success status
                'error': Error message if any (None if success)
        """
        result = {
            "emotions": {},
            "dominant_emotion": self.default_emotion,
            "success": False,
            "error": None
        }
        
        if not text:
            result["error"] = "Empty text provided"
            return result
        
        try:
            # Normalize and prepare the text
            text_lower = text.lower()
            
            # Count emotion keywords
            emotion_counts = {emotion: 0 for emotion in self.emotion_keywords}
            
            for emotion, keywords in self.emotion_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        emotion_counts[emotion] += 1
            
            # Calculate simple confidence scores (normalized by keyword count)
            total_matches = sum(emotion_counts.values())
            
            if total_matches > 0:
                # Create normalized confidence scores
                emotion_scores = {
                    emotion: count / total_matches 
                    for emotion, count in emotion_counts.items() 
                    if count > 0
                }
                
                # Add neutral if no emotions detected
                if not emotion_scores:
                    emotion_scores["neutral"] = 1.0
                
                # Get dominant emotion
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                
                # Update result
                result["emotions"] = emotion_scores
                result["dominant_emotion"] = dominant_emotion
                result["success"] = True
            else:
                # No emotions detected, use neutral as default
                result["emotions"] = {"neutral": 1.0}
                result["dominant_emotion"] = "neutral"
                result["success"] = True
            
        except Exception as e:
            result["error"] = f"Error during emotion detection: {e}"
            
        return result
    
    def get_available_emotions(self) -> List[str]:
        """Get the list of emotions that this detector can recognize
        
        Returns:
            List of emotion names that this detector can identify
        """
        return list(self.emotion_keywords.keys())


class DashscopeEmotionDetector(EmotionDetector):
    """Emotion detection using Dashscope NLP capabilities
    
    This implementation uses Dashscope's NLP capabilities to analyze
    text sentiment and map it to emotional states.
    """
    
    def __init__(self, api_key=None):
        """Initialize the Dashscope emotion detector
        
        Args:
            api_key: Optional Dashscope API key
        """
        import os
        from dotenv import load_dotenv
        import json
        
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
        
        try:
            import dashscope
            # Configure DashScope
            dashscope.api_key = self.api_key
        except ImportError:
            print("Warning: dashscope module not found. Please install with pip install dashscope")
        
        # Define emotions this detector can identify
        self.available_emotions = [
            "happy", "sad", "angry", "afraid",
            "surprised", "disgusted", "neutral"
        ]
        
        # Default emotion when no emotions are detected
        self.default_emotion = "neutral"
        
    def detect_emotion_from_audio(self, audio_data: Any) -> Dict[str, Any]:
        """Detect emotions from audio input (not implemented)
        
        Args:
            audio_data: Audio data to analyze
            
        Returns:
            Dict with result indicating not implemented
        """
        return {
            "emotions": {},
            "dominant_emotion": self.default_emotion,
            "success": False,
            "error": "Audio-based emotion detection not implemented"
        }
    
    def detect_emotion_from_text(self, text: str) -> Dict[str, Any]:
        """Detect emotions from text using Dashscope
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with emotion detection results
        """
        result = {
            "emotions": {},
            "dominant_emotion": self.default_emotion,
            "success": False,
            "error": None
        }
        
        if not text:
            result["error"] = "Empty text provided"
            return result
        
        try:
            import dashscope
            from dashscope.nlp.sentiment_analysis import SentimentAnalysis
            
            # Call the sentiment analysis API
            response = SentimentAnalysis.call(
                model="sentiment-analysis-bilingual",
                input={"text": text}
            )
            
            if response.status_code == 200:
                # Extract sentiment information
                sentiment = response.output.get("result", {})
                sentiment_label = sentiment.get("label", "neutral").lower()
                sentiment_score = sentiment.get("score", 0.0)
                
                # Map DashScope sentiment to our emotion categories
                if sentiment_label == "positive":
                    dominant_emotion = "happy"
                    emotions = {
                        "happy": sentiment_score,
                        "neutral": 1.0 - sentiment_score
                    }
                elif sentiment_label == "negative":
                    # For negative, try to determine if it's sad or angry
                    # This is a simplification - more sophisticated analysis would be better
                    if "sad" in text.lower() or "unhappy" in text.lower():
                        dominant_emotion = "sad"
                        emotions = {
                            "sad": sentiment_score,
                            "neutral": 1.0 - sentiment_score
                        }
                    else:
                        dominant_emotion = "angry"
                        emotions = {
                            "angry": sentiment_score,
                            "neutral": 1.0 - sentiment_score
                        }
                else:
                    dominant_emotion = "neutral"
                    emotions = {"neutral": 1.0}
                
                result["emotions"] = emotions
                result["dominant_emotion"] = dominant_emotion
                result["success"] = True
                
            else:
                # API call failed, fall back to keyword detection
                print(f"DashScope API error: {response.code} - {response.message}")
                fallback = TextBasedEmotionDetector()
                fallback_result = fallback.detect_emotion_from_text(text)
                
                result["emotions"] = fallback_result["emotions"]
                result["dominant_emotion"] = fallback_result["dominant_emotion"]
                result["success"] = True
                result["error"] = f"Used fallback due to API error: {response.code}"
                
        except Exception as e:
            result["error"] = f"Error during emotion detection: {e}"
            
            # Try fallback if DashScope fails
            try:
                fallback = TextBasedEmotionDetector()
                fallback_result = fallback.detect_emotion_from_text(text)
                
                result["emotions"] = fallback_result["emotions"]
                result["dominant_emotion"] = fallback_result["dominant_emotion"]
                result["success"] = True
                result["error"] = f"Used fallback due to error: {e}"
            except:
                pass
                
        return result
    
    def get_available_emotions(self) -> List[str]:
        """Get the list of emotions that this detector can recognize
        
        Returns:
            List of emotion names that this detector can identify
        """
        return self.available_emotions