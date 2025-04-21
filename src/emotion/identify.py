import torch
import cv2
import time
import os
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import threading
from typing import Dict, Any, Optional, List, Callable

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))  # This will reduce width and height by half
    return nn.Sequential(*blk)


def vgg(conv_arch, fc_features, fc_hidden_units):
    net = nn.Sequential()
    # Convolutional layers part
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # Each vgg_block reduces width and height by half
        net.add_module("vgg_block_" + str(i + 1), vgg_block(num_convs, in_channels, out_channels))
    # Fully connected layers part
    net.add_module("fc", nn.Sequential(
        Reshape(),
        nn.Linear(fc_features, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, 7)
    ))
    return net


class EmotionDetectorCamera:
    """Camera-based emotion detection class that provides real-time emotion detection"""
    
    # Emotion categories
    EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # Chinese emotion categories (optional)
    EMOTION_CLASSES_ZH = ['愤怒', '厌恶', '恐惧', '开心', '悲伤', '惊讶', '平静']
    
    # Modified conv_arch definition
    CONV_ARCH = ((1, 1, 32), (1, 32, 64), (2, 64, 128))  # Change first channel to 1 (grayscale)
    FC_FEATURES = 128 * 6 * 6  # May need to be adjusted based on input size
    FC_HIDDEN_UNITS = 1024
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 detection_interval: float = 0.5,
                 use_chinese: bool = False,
                 callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize camera emotion detector
        
        Args:
            model_path: Model path, use default path if None
            detection_interval: Detection time interval (seconds)
            use_chinese: Whether to use Chinese emotion labels
            callback: Callback function when emotion is detected
        """
        # Set default model path
        if model_path is None:
            # Try to find the model file in the project root directory
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(root_dir, "best_model.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.detection_interval = detection_interval
        self.callback = callback
        self.emotion_classes = self.EMOTION_CLASSES_ZH if use_chinese else self.EMOTION_CLASSES
        
        # Initialize model
        self.model = self._load_model()
        
        # Initialize camera
        self.cap = None
        
        # Thread and control flags
        self.detection_thread = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Latest detection result
        self.latest_result = {
            "emotion": "neutral",
            "emotion_index": 6,  # Index of Neutral
            "probability": 0.0,
            "all_probabilities": {emotion: 0.0 for emotion in self.emotion_classes},
            "timestamp": time.time()
        }
        
        print(f"Camera emotion detector initialized, using device: {device}")
    
    def _load_model(self):
        """Load model"""
        try:
            model = vgg(self.CONV_ARCH, self.FC_FEATURES, self.FC_HIDDEN_UNITS)
            model.load_state_dict(torch.load(self.model_path, map_location=device))
            model = model.to(device)
            model.eval()
            print(f"Emotion detection model loaded successfully: {self.model_path}")
            return model
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            raise
    
    def _preprocess_image(self, image):
        """Image preprocessing - Convert to grayscale and normalize"""
        # Convert BGR to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),  # Adjust according to model input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale only needs one channel mean and std
        ])
        return transform(gray_image).unsqueeze(0).to(device)
    
    def start(self, camera_id: int = 0, show_video: bool = False):
        """
        Start emotion detection
        
        Args:
            camera_id: Camera ID, default is 0 (usually built-in camera)
            show_video: Whether to display video window
            
        Returns:
            Whether successfully started
        """
        if self.is_running:
            print("Emotion detection is already running")
            return False
        
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                print(f"Unable to open camera ID: {camera_id}")
                return False
            
            # Set status and start thread
            self.is_running = True
            self.detection_thread = threading.Thread(
                target=self._detection_loop,
                args=(show_video,),
                daemon=True
            )
            self.detection_thread.start()
            print(f"Emotion detection started successfully, using camera ID: {camera_id}")
            return True
            
        except Exception as e:
            print(f"Failed to start emotion detection: {str(e)}")
            self.is_running = False
            if self.cap is not None:
                self.cap.release()
            return False
    
    def stop(self):
        """Stop emotion detection"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for thread to end
        if self.detection_thread is not None and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
        
        # Release camera
        if self.cap is not None:
            self.cap.release()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print("Emotion detection stopped")
    
    def get_latest_emotion(self) -> Dict[str, Any]:
        """
        Get the latest emotion detection result
        
        Returns:
            Dict containing emotion detection result
        """
        with self.lock:
            return self.latest_result.copy()
    
    def _detection_loop(self, show_video: bool):
        """
        Emotion detection main loop
        
        Args:
            show_video: Whether to display video window
        """
        last_detection_time = 0
        
        while self.is_running and self.cap is not None:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Unable to get video frame")
                break
            
            # If video display is needed
            if show_video:
                # Display detection results on video
                with self.lock:
                    emotion = self.latest_result["emotion"]
                    prob = self.latest_result["probability"]
                    
                # Draw text on frame
                text = f"{emotion}: {prob*100:.1f}%" if prob > 0 else "Detecting..."
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Display video
                cv2.imshow('Emotion Detection', frame)
                
                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    break
            
            # Check if detection interval is reached
            current_time = time.time()
            if current_time - last_detection_time >= self.detection_interval:
                try:
                    # Detect emotion
                    self._detect_emotion(frame)
                    last_detection_time = current_time
                except Exception as e:
                    print(f"Emotion detection error: {str(e)}")
        
        # Release resources when ending
        if self.cap is not None:
            self.cap.release()
        if show_video:
            cv2.destroyAllWindows()
    
    def _detect_emotion(self, frame):
        """
        Detect emotion for a single frame
        
        Args:
            frame: Video frame
        """
        try:
            # Preprocess image
            input_image = self._preprocess_image(frame)
            
            # Predict emotion
            with torch.no_grad():
                outputs = self.model(input_image)
                # Calculate softmax probabilities
                probabilities = F.softmax(outputs, dim=1)
                # Get prediction results and probabilities
                probs, predicted = torch.max(probabilities, 1)
                emotion_index = predicted.item()
                emotion = self.emotion_classes[emotion_index]
                prob_value = probs.item()
                
                # Get probabilities for all categories
                prob_list = probabilities.squeeze().cpu().numpy()
                all_probs = {self.emotion_classes[i]: float(prob_list[i]) 
                            for i in range(len(self.emotion_classes))}
            
            # Update latest result
            with self.lock:
                self.latest_result = {
                    "emotion": emotion,
                    "emotion_index": emotion_index,
                    "probability": prob_value,
                    "all_probabilities": all_probs,
                    "timestamp": time.time()
                }
            
            # Call callback function (if any)
            if self.callback is not None:
                self.callback(self.latest_result)
            
        except Exception as e:
            print(f"Emotion detection processing error: {str(e)}")
            raise


# Example usage when running this script directly
if __name__ == "__main__":
    # Define callback function
    def print_emotion(result):
        print(f"\nDetection time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Emotion: {result['emotion']} (Confidence: {result['probability']*100:.2f}%)")
        print("Probabilities for all categories:")
        for emo, prob in result['all_probabilities'].items():
            print(f"  {emo}: {prob*100:.2f}%")
    
    # Create detector instance
    try:
        detector = EmotionDetectorCamera(
            detection_interval=1.0,  # Detect once per second
            callback=print_emotion,  # Set callback function
            use_chinese=True         # Use Chinese emotion labels
        )
        
        # Start detection, show video window
        if detector.start(show_video=True):
            print("Press 'q' to stop detection")
            
            # Main thread waits
            try:
                while detector.is_running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Detection interrupted by user")
            finally:
                detector.stop()
    
    except Exception as e:
        print(f"Program error: {str(e)}")