# Updated identify.py using DeepFace instead of dlib 68 landmarks

import torch
import cv2
import time
import os
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import threading
from typing import Dict, Any, Optional, Callable
from PIL import ImageFont, ImageDraw, Image
import collections
from deepface import DeepFace

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 多帧投票窗口
VOTE_WINDOW = 5
emotion_window = collections.deque(maxlen=VOTE_WINDOW)

# 表情偏置权重
bias_weights = {
    "happy": 1,
    "neutral": 0.5,
    "sad": 2,
    "angry": 0.9,
    "fear": 1.0,
    "disgust": 1.0,
    "surprise": 1.6
}


def cv2_putText_cn(img, text, position, font_path="simhei.ttf", font_size=32, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        raise RuntimeError(f"字体加载失败：{e}")
    draw.text(position, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class EmotionDetectorCamera:
    EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    EMOTION_CLASSES_ZH = ['愤怒', '厌恶', '恐惧', '开心', '悲伤', '惊讶', '平静']

    def __init__(self,
                 detection_interval: float = 0.5,
                 use_chinese: bool = False,
                 callback: Optional[Callable[[Dict[str, Any]], None]] = None):

        self.detection_interval = detection_interval
        self.callback = callback
        self.emotion_classes = self.EMOTION_CLASSES_ZH if use_chinese else self.EMOTION_CLASSES
        self.cap = None
        self.detection_thread = None
        self.is_running = False
        self.lock = threading.Lock()
        self.latest_result = {
            "emotion": "neutral",
            "emotion_index": 6,
            "probability": 0.0,
            "all_probabilities": {emotion: 0.0 for emotion in self.emotion_classes},
            "timestamp": time.time()
        }

        print(f"Camera emotion detector initialized, using device: {device}")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 480)

        self.TEMP_IMG_PATH = "temp_frame.jpg"
        self.last_valid_face_rect = None
        self.demographics = {"age": None, "gender": None, "gender_confidence": None}
        self.demographics_initialized = False

    def show_text(self, frame, text, position=(50, 50), color=(0, 255, 0), size=1.0):
        if self.emotion_classes == self.EMOTION_CLASSES_ZH:
            frame = cv2_putText_cn(frame, text, position, font_size=int(32 * size), color=color)
        else:
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)
        return frame

    def start(self, camera_id: int = 0, show_video: bool = False):
        if self.is_running:
            print("Emotion detection is already running")
            return False

        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                print(f"Unable to open camera ID: {camera_id}")
                return False

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
        if not self.is_running:
            return

        self.is_running = False
        if self.detection_thread is not None and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Emotion detection stopped")

    def get_latest_emotion(self) -> Dict[str, Any]:
        with self.lock:
            return self.latest_result.copy()

    def _detection_loop(self, show_video: bool):
        last_detection_time = 0
        current_emotion = "neutral"

        while self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print("Unable to get video frame")
                break

            current_time = time.time()
            display_frame = frame.copy()

            if self.last_valid_face_rect:
                x, y, w, h = self.last_valid_face_rect
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if current_time - last_detection_time >= self.detection_interval:
                try:
                    cv2.imwrite(self.TEMP_IMG_PATH, frame)
                    face_objs = DeepFace.extract_faces(
                        img_path=self.TEMP_IMG_PATH,
                        detector_backend="ssd",
                        enforce_detection=False,
                        align=False
                    )

                    if face_objs:
                        main_face = max(face_objs, key=lambda x: x["facial_area"]["w"] * x["facial_area"]["h"])
                        face_area = main_face["facial_area"]
                        self.last_valid_face_rect = (face_area["x"], face_area["y"], face_area["w"], face_area["h"])

                        face_img = frame[face_area["y"]:face_area["y"] + face_area["h"],
                                         face_area["x"]:face_area["x"] + face_area["w"]]
                        cv2.imwrite(self.TEMP_IMG_PATH, face_img)

                        if not self.demographics_initialized:
                            results = DeepFace.analyze(
                                img_path=self.TEMP_IMG_PATH,
                                actions=["emotion", "age", "gender"],
                                detector_backend="skip",
                                enforce_detection=False,
                                silent=True
                            )
                            if results:
                                r = results[0]
                                self.demographics["age"] = int(r["age"])
                                self.demographics["gender"] = r["dominant_gender"]
                                self.demographics["gender_confidence"] = r["gender"][r["dominant_gender"]]
                                self.demographics_initialized = True

                                raw_emotions = r["emotion"]
                                biased_emotions = {emo: raw_emotions[emo] * bias_weights.get(emo, 1.0) for emo in raw_emotions}
                                emotion_window.append(biased_emotions)
                                current_emotion = max(biased_emotions, key=biased_emotions.get)
                        else:
                            results = DeepFace.analyze(
                                img_path=self.TEMP_IMG_PATH,
                                actions=["emotion"],
                                detector_backend="skip",
                                enforce_detection=False,
                                silent=True
                            )
                            if results:
                                raw_emotions = results[0]["emotion"]
                                biased_emotions = {emo: raw_emotions[emo] * bias_weights.get(emo, 1.0) for emo in raw_emotions}
                                emotion_window.append(biased_emotions)

                                combined_scores = {}
                                for e in emotion_window:
                                    for emo, score in e.items():
                                        combined_scores[emo] = combined_scores.get(emo, 0) + score
                                for emo in combined_scores:
                                    combined_scores[emo] /= len(emotion_window)

                                current_emotion = max(combined_scores, key=combined_scores.get)

                        with self.lock:
                            self.latest_result = {
                                "emotion": current_emotion,
                                "emotion_index": self.EMOTION_CLASSES.index(current_emotion.capitalize()) if current_emotion.capitalize() in self.EMOTION_CLASSES else 6,
                                "probability": 1.0,
                                "all_probabilities": {emo: 1.0 if emo.lower() == current_emotion.lower() else 0.0 for emo in self.emotion_classes},
                                "timestamp": time.time()
                            }
                            if self.callback:
                                self.callback(self.latest_result)

                        last_detection_time = current_time

                    if os.path.exists(self.TEMP_IMG_PATH):
                        os.remove(self.TEMP_IMG_PATH)

                except Exception as e:
                    print(f"检测失败: {e}")
                    if os.path.exists(self.TEMP_IMG_PATH):
                        os.remove(self.TEMP_IMG_PATH)

            display_frame = self.show_text(display_frame, f"Emotion: {current_emotion}", (10, 30), (0, 255, 0), 0.9)

            if self.last_valid_face_rect and self.demographics["age"] is not None:
                x, y, w, h = self.last_valid_face_rect
                gender_text = f"{self.demographics['gender']}"
                display_frame = self.show_text(display_frame, f"Age: {self.demographics['age']}", (x, y - 50), (0, 255, 0), 0.6)
                display_frame = self.show_text(display_frame, gender_text, (x, y - 20), (0, 255, 0), 0.6)

            if show_video:
                cv2.imshow('Emotion Detection', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    break

        if self.cap is not None:
            self.cap.release()
        if show_video:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    def print_emotion(result):
        print(f"\nDetection time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Emotion: {result['emotion']} (Confidence: {result['probability'] * 100:.2f}%)")
        print("Probabilities for all categories:")
        for emo, prob in result['all_probabilities'].items():
            print(f"  {emo}: {prob * 100:.2f}%")

    try:
        detector = EmotionDetectorCamera(
            detection_interval=0.5,
            callback=print_emotion,
            use_chinese=False
        )
        if detector.start(show_video=True):
            print("Press 'q' to stop detection")
            while detector.is_running:
                time.sleep(0.1)
            detector.stop()
    except Exception as e:
        print(f"Program error: {str(e)}")
