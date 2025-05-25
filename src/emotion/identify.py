# Updated identify.py with 5-second stable heart rate display

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
from scipy.signal import find_peaks, butter, filtfilt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 多帧投票窗口
VOTE_WINDOW = 5
emotion_window = collections.deque(maxlen=VOTE_WINDOW)

# 表情偏置权重
bias_weights = {
    "happy": 1,
    "neutral": 1,
    "sad": 1.5,
    "angry": 0.7,
    "fear": 1.0,
    "disgust": 1.0,
    "surprise": 1.6
}

# 心跳检测参数
HR_FPS = 30
HR_WINDOW_SIZE = HR_FPS * 5  # 5秒数据
HR_UPDATE_INTERVAL = 3.0  # 每5秒更新一次心率
HR_SMOOTHING_WINDOW = 3  # 心率平滑窗口大小


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

        # 心跳检测相关变量
        self.hr_signal_buffer = []
        self.last_hr_update_time = 0
        self.locked_hr_value = None  # 锁定的心率值
        self.is_hr_calculating = False
        self.hr_display = "Calculating HR..."
        self.hr_history = []  # 心率历史记录用于平滑

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
            result = self.latest_result.copy()
            result["heart_rate"] = self.locked_hr_value
            return result

    def _butter_bandpass_filter(self, data, lowcut=0.7, highcut=4.0, fs=30, order=5):
        """带通滤波器，用于心率信号处理"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def _update_hr_value(self):
        """在子线程中计算心率"""
        try:
            if len(self.hr_signal_buffer) == HR_WINDOW_SIZE:
                filtered = self._butter_bandpass_filter(self.hr_signal_buffer)
                peaks, _ = find_peaks(filtered, distance=HR_FPS * 0.6)  # 至少间隔0.6秒

                if len(peaks) >= 2:
                    new_hr = 60 / (np.mean(np.diff(peaks)) / HR_FPS)
                    self.hr_history.append(new_hr)

                    # 平滑处理
                    if len(self.hr_history) > HR_SMOOTHING_WINDOW:
                        self.hr_history.pop(0)

                    self.locked_hr_value = np.mean(self.hr_history)
                    self.hr_display = f"HR: {self.locked_hr_value:.1f} BPM"
        finally:
            self.is_hr_calculating = False
            self.last_hr_update_time = time.time()
            self.hr_signal_buffer = []  # 重置缓冲区

    def _collect_hr_data(self, frame):
        """采集心率数据但不计算"""
        if self.last_valid_face_rect and not self.is_hr_calculating:
            x, y, w, h = self.last_valid_face_rect
            # 使用面部中央区域提高稳定性
            roi = frame[y + h // 4:y + h * 3 // 4, x + w // 4:x + w * 3 // 4]
            self.hr_signal_buffer.append(np.mean(roi[:, :, 1]))  # 绿色通道

            # 当缓冲区满且到5秒间隔时触发计算
            current_time = time.time()
            if (len(self.hr_signal_buffer) >= HR_WINDOW_SIZE and
                    current_time - self.last_hr_update_time >= HR_UPDATE_INTERVAL):
                self.is_hr_calculating = True
                threading.Thread(target=self._update_hr_value, daemon=True).start()

    def _detection_loop(self, show_video: bool):
        last_detection_time = 0
        current_emotion = "neutral"

        # 显示参数配置
        font_scale = 0.6  # 字体大小
        text_color = (0, 255, 0)  # 文字颜色(绿色)
        text_thickness = 1  # 文字粗细
        line_height = 25  # 行间距
        start_y = 20  # 起始y坐标
        text_x = 10  # 起始x坐标

        while self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print("Unable to get video frame")
                break

            current_time = time.time()
            display_frame = frame.copy()

            # 1. 采集心率数据（每帧都执行）
            self._collect_hr_data(frame)

            # 2. 直接在画面上绘制信息（无背景条）
            y_pos = display_frame.shape[0] - 80  # 从底部向上定位

            # 第一行：表情
            cv2.putText(display_frame, f"Emotion: {current_emotion}",
                        (text_x, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)

            # 第二行：心率（强制转换为整数）
            hr_value = int(self.locked_hr_value) if self.locked_hr_value is not None else "Calculating"
            cv2.putText(display_frame, f"Heart Rate: {hr_value}",
                        (text_x, y_pos + line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)

            # 第三行：人口统计信息（如果有）
            if self.demographics["age"] is not None:
                demo_text = f"Age: {self.demographics['age']}  Gender: {self.demographics['gender']}"
                cv2.putText(display_frame, demo_text,
                            (text_x, y_pos + 2 * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)

            # 3. 人脸检测和绘制
            if self.last_valid_face_rect:
                x, y, w_rect, h_rect = self.last_valid_face_rect
                cv2.rectangle(display_frame, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 2)

            # 4. 定时执行表情检测（保持原有逻辑）
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
                                "timestamp": time.time(),
                                "heart_rate": int(self.locked_hr_value) if self.locked_hr_value is not None else None
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

            # 5. 显示画面
            if show_video:
                cv2.imshow('Emotion & Heart Rate Detection', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    break

        # 清理资源
        if self.cap is not None:
            self.cap.release()
        if show_video:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    def print_result(result):
        print(f"\nDetection time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Emotion: {result['emotion']}")
        if result.get('heart_rate') is not None:
            print(f"Heart Rate: {result['heart_rate']:.1f} BPM")
        else:
            print("Heart Rate: Calculating...")
        print("Probabilities for all categories:")
        for emo, prob in result['all_probabilities'].items():
            print(f"  {emo}: {prob * 100:.2f}%")


    try:
        detector = EmotionDetectorCamera(
            detection_interval=0.5,
            callback=print_result,
            use_chinese=False
        )
        if detector.start(show_video=True):
            print("Press 'q' to stop detection")
            while detector.is_running:
                time.sleep(0.1)
            detector.stop()
    except Exception as e:
        print(f"Program error: {str(e)}")