# Updated identify.py with 68-point facial landmark detection for micro-expression analysis

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
import dlib

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize dlib's face detector and landmark predictor
detector_dlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/57124/Downloads/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

# 多帧投票窗口
VOTE_WINDOW = 5
emotion_window = collections.deque(maxlen=VOTE_WINDOW)
micro_expression_window = collections.deque(maxlen=VOTE_WINDOW)

# 表情偏置权重
bias_weights = {
    "happy": 2,
    "neutral": 1,
    "sad": 1.3,
    "angry": 0.8,
    "fear": 1.0,
    "disgust": 1.0,
    "surprise": 1.6
}

# 阈值设置
thresholds = {
    'mouth_higth_happy': 0.03,
    'mouth_higth_amazing': 0.025,
    'eye_hight_amazing': 0.045,
    'brow_k_angry': -0.1,
    'mouth_higth_sad': -0.01,
    'brow_width_disgust': 0.5,
    'eye_hight_fear': 0.05,
    'mouth_core_width': 0.068,
    'mouth_core_hight': 0.0055,
    'eye_core_hight': 0.23
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


def extract_features(landmarks, face_rect):
    face_width = face_rect.right() - face_rect.left()
    face_height = face_rect.bottom() - face_rect.top()

    # 嘴巴特征
    mouth_width = (landmarks.part(54).x - landmarks.part(48).x) / face_width
    mouth_higth = (landmarks.part(51).y - landmarks.part(57).y) / face_width
    mouth_core_width = (landmarks.part(59).x - landmarks.part(48).x) / face_width
    mouth_core_hight = (landmarks.part(48).y - landmarks.part(60).y) / face_width
    eye_core_hight = (landmarks.part(42).y - landmarks.part(22).y) / face_width

    # 眉毛特征
    brow_sum = 0  # 高度之和
    frown_sum = 0  # 两边眉毛距离之和
    line_brow_x = []
    line_brow_y = []

    for j in range(17, 21):
        brow_sum += (landmarks.part(j).y - face_rect.top()) + (landmarks.part(j + 5).y - face_rect.top())
        frown_sum += landmarks.part(j + 5).x - landmarks.part(j).x
        line_brow_x.append(landmarks.part(j).x)
        line_brow_y.append(landmarks.part(j).y)

    # 眉毛倾斜度
    tempx = np.array(line_brow_x)
    tempy = np.array(line_brow_y)
    if len(tempx) > 0 and len(tempy) > 0:
        z1 = np.polyfit(tempx, tempy, 1)
        brow_k = -round(z1[0], 3)
    else:
        brow_k = 0

    brow_hight = (brow_sum / 10) / face_width  # 眉毛高度占比
    brow_width = (frown_sum / 5) / face_width  # 眉毛距离占比

    # 眼睛睁开程度
    eye_sum = (landmarks.part(41).y - landmarks.part(37).y + landmarks.part(40).y - landmarks.part(38).y +
               landmarks.part(47).y - landmarks.part(43).y + landmarks.part(46).y - landmarks.part(44).y)
    eye_hight = (eye_sum / 4) / face_width

    # 鼻子皱起程度 (厌恶表情)
    nose_wrinkling = (landmarks.part(31).y - landmarks.part(27).y) / face_height

    return {
        'mouth_width': mouth_width,
        'mouth_higth': mouth_higth,
        'brow_k': brow_k,
        'brow_hight': brow_hight,
        'brow_width': brow_width,
        'eye_hight': eye_hight,
        'nose_wrinkling': nose_wrinkling,
        'mouth_core_width': mouth_core_width,
        'mouth_core_hight': mouth_core_hight,
        'eye_core_hight': eye_core_hight
    }


def classify_micro_expression(features):
    # 惊讶 (眼睛睁大+嘴巴张大)
    if (features['eye_core_hight'] >= thresholds['eye_core_hight']):
        return "惊讶"
    elif (features['mouth_higth'] >= thresholds['mouth_higth_amazing'] and
          features['eye_hight'] >= thresholds['eye_hight_amazing']):
        return "惊讶"

    # 开心 (嘴巴张大但眼睛不一定)
    elif features['mouth_core_width'] >= thresholds['mouth_core_width']:
        return "开心"

    # 愤怒 (眉毛内聚且下压)
    elif features['brow_k'] <= thresholds['brow_k_angry']:
        return "愤怒"

    # 悲伤 (眉毛外角上扬)
    elif features['mouth_core_hight'] >= thresholds['mouth_core_hight']:
        return "悲伤"

    # 厌恶 (鼻子皱起+眉毛压低)
    elif (features['nose_wrinkling'] < 0.25):
        return "厌恶"

    # 恐惧 (眼睛睁大+眉毛上扬)
    elif (features['eye_hight'] >= thresholds['eye_hight_fear'] and
          features['brow_hight'] > 0.25):
        return "恐惧"

    # 默认自然表情
    else:
        return "自然"


def get_combined_micro_expression():
    if not micro_expression_window:
        return "自然"

    counts = {}
    for me in micro_expression_window:
        counts[me] = counts.get(me, 0) + 1

    return max(counts, key=counts.get)


def get_refined_emotion(main_emotion, micro_expression):
    # 主表情与微表情的修正规则
    combination_rules = {
        # 中性表情组合
        "neutral": {
            "开心": "微笑",
            "愤怒": "严肃",
            "悲伤": "忧郁",
            "惊讶": "好奇",
            "厌恶": "不适",
            "恐惧": "不安",
            "自然": "中性"
        },
        # 开心组合
        "happy": {
            "开心": "开怀大笑",
            "愤怒": "假笑",
            "悲伤": "苦笑",
            "惊讶": "惊喜",
            "厌恶": "嘲讽",
            "恐惧": "紧张的笑",
            "自然": "微笑"
        },
        # 悲伤组合
        "sad": {
            "开心": "喜极而泣",
            "愤怒": "愤懑",
            "悲伤": "悲痛",
            "惊讶": "震惊的悲伤",
            "厌恶": "轻蔑",
            "恐惧": "绝望",
            "自然": "忧郁"
        },
        # 愤怒组合
        "angry": {
            "开心": "狞笑",
            "愤怒": "暴怒",
            "悲伤": "愤懑",
            "惊讶": "震怒",
            "厌恶": "憎恶",
            "恐惧": "威胁",
            "自然": "不悦"
        },
        # 惊讶组合
        "surprise": {
            "开心": "惊喜",
            "愤怒": "震惊的愤怒",
            "悲伤": "震惊的悲伤",
            "惊讶": "极度惊讶",
            "厌恶": "震惊的厌恶",
            "恐惧": "惊恐",
            "自然": "惊讶"
        },
        # 厌恶组合
        "disgust": {
            "开心": "讥笑",
            "愤怒": "憎恶",
            "悲伤": "厌恶的悲伤",
            "惊讶": "震惊的厌恶",
            "厌恶": "极度厌恶",
            "恐惧": "恶心",
            "自然": "轻微厌恶"
        },
        # 恐惧组合
        "fear": {
            "开心": "紧张的笑",
            "愤怒": "恐惧的愤怒",
            "悲伤": "恐惧的悲伤",
            "惊讶": "惊恐",
            "厌恶": "恐惧的厌恶",
            "恐惧": "极度恐惧",
            "自然": "不安"
        }
    }

    # 确保主表情在规则中
    main_emotion_lower = main_emotion.lower()
    if main_emotion_lower not in combination_rules:
        return main_emotion

    # 获取对应规则
    rules = combination_rules[main_emotion_lower]

    # 如果微表情在规则中，返回组合结果
    if micro_expression in rules:
        return rules[micro_expression]

    # 默认返回主表情
    return main_emotion


def get_confidence(main_emotion, micro_expression):
    # 如果两种结果一致，置信度高
    if (main_emotion == "happy" and micro_expression == "开心") or \
            (main_emotion == "angry" and micro_expression == "愤怒") or \
            (main_emotion == "sad" and micro_expression == "悲伤") or \
            (main_emotion == "surprise" and micro_expression == "惊讶") or \
            (main_emotion == "disgust" and micro_expression == "厌恶") or \
            (main_emotion == "fear" and micro_expression == "恐惧"):
        return "高置信度"

    # 如果微表情是自然，使用主表情
    if micro_expression == "自然":
        return "中等置信度"

    # 其他情况为中等或低置信度
    return "低置信度"


class EmotionDetectorCamera:
    EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    EMOTION_CLASSES_ZH = ['愤怒', '厌恶', '恐惧', '开心', '悲伤', '惊讶', '平静']

    def __init__(self,
                 detection_interval: float = 0.5,
                 use_chinese: bool = False,
                 callback: Optional[Callable[[Dict[str, Any]], None]] = None):

        self.detection_interval = detection_interval
        self.use_chinese = use_chinese  # 添加这一行
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
            "timestamp": time.time(),
            "refined_emotion": "neutral",
            "confidence": "高置信度" if use_chinese else "High confidence"
        }

        # 心跳检测相关变量
        self.hr_signal_buffer = []
        self.last_hr_update_time = 0
        self.locked_hr_value = None
        self.is_hr_calculating = False
        self.hr_display = "Calculating HR..."
        self.hr_history = []

        # 字体路径检查
        self.font_path = "simhei.ttf"
        if use_chinese and not os.path.exists(self.font_path):
            print(f"警告: 中文字体文件 {self.font_path} 不存在，将回退到英文显示")
            self.use_chinese = False

        print(f"Camera emotion detector initialized, using device: {device}")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 480)

        self.TEMP_IMG_PATH = "temp_frame.jpg"
        self.last_valid_face_rect = None
        self.demographics = {
            "age": None,
            "gender": None,
            "gender_confidence": None,
            "race": None,
            "race_confidence": None
        }
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
        current_micro = "neutral"
        refined_emotion = "neutral"
        confidence = "High confidence"

        # Display parameters
        font_scale = 0.6
        text_color = (0, 0, 255)  # Red text
        text_thickness = 1
        line_height = 25
        y_pos = 20  # Display at top
        text_x = 10

        # English version of combination rules
        combination_rules = {
            "neutral": {
                "happy": "Slight Smile",
                "angry": "Serious",
                "sad": "Melancholy",
                "surprise": "Curious",
                "disgust": "Discomfort",
                "fear": "Uneasy",
                "neutral": "Neutral"
            },
            "happy": {
                "happy": "Laughing",
                "angry": "Fake Smile",
                "sad": "Bitter Smile",
                "surprise": "Pleasantly Surprised",
                "disgust": "Mocking",
                "fear": "Nervous Smile",
                "neutral": "Smiling"
            },
            "sad": {
                "happy": "Tears of Joy",
                "angry": "Resentful",
                "sad": "Grief",
                "surprise": "Shocked Sadness",
                "disgust": "Scorn",
                "fear": "Despair",
                "neutral": "Melancholy"
            },
            "angry": {
                "happy": "Grin",
                "angry": "Furious",
                "sad": "Resentful",
                "surprise": "Outraged",
                "disgust": "Loathing",
                "fear": "Threatening",
                "neutral": "Displeased"
            },
            "surprise": {
                "happy": "Delighted",
                "angry": "Shocked Anger",
                "sad": "Shocked Sadness",
                "surprise": "Astonished",
                "disgust": "Shocked Disgust",
                "fear": "Terrified",
                "neutral": "Surprised"
            },
            "disgust": {
                "happy": "Sneer",
                "angry": "Loathing",
                "sad": "Disgusted Sadness",
                "surprise": "Shocked Disgust",
                "disgust": "Revolted",
                "fear": "Nauseated",
                "neutral": "Mild Disgust"
            },
            "fear": {
                "happy": "Nervous Smile",
                "angry": "Fearful Anger",
                "sad": "Fearful Sadness",
                "surprise": "Terrified",
                "disgust": "Fearful Disgust",
                "fear": "Terror",
                "neutral": "Uneasy"
            }
        }

        def get_refined_emotion_en(main_emotion, micro_expression):
            main_emotion_lower = main_emotion.lower()
            if main_emotion_lower not in combination_rules:
                return main_emotion
            return combination_rules[main_emotion_lower].get(micro_expression, main_emotion)

        def get_confidence_en(main_emotion, micro_expression):
            if (main_emotion == "happy" and micro_expression == "happy") or \
                    (main_emotion == "angry" and micro_expression == "angry") or \
                    (main_emotion == "sad" and micro_expression == "sad") or \
                    (main_emotion == "surprise" and micro_expression == "surprise") or \
                    (main_emotion == "disgust" and micro_expression == "disgust") or \
                    (main_emotion == "fear" and micro_expression == "fear"):
                return "High confidence"
            if micro_expression == "neutral":
                return "Medium confidence"
            return "Low confidence"

        while self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to get video frame")
                break

            current_time = time.time()
            display_frame = frame.copy()

            # 1. Collect HR data (every frame)
            self._collect_hr_data(frame)

            # 2. Display information (top of frame)
            hr_value = int(self.locked_hr_value) if self.locked_hr_value is not None else "Calculating"

            cv2.putText(display_frame, f"Emotion: {current_emotion}",
                        (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)
            cv2.putText(display_frame, f"Micro: {current_micro}",
                        (text_x, y_pos + line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)
            cv2.putText(display_frame, f"Refined: {refined_emotion}",
                        (text_x, y_pos + 2 * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)
            cv2.putText(display_frame, f"Confidence: {confidence}",
                        (text_x, y_pos + 3 * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)
            cv2.putText(display_frame, f"Heart Rate: {hr_value}",
                        (text_x, y_pos + 4 * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)

            # 3. Face detection and drawing
            if self.last_valid_face_rect:
                x, y, w_rect, h_rect = self.last_valid_face_rect
                cv2.rectangle(display_frame, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 2)

            # 4. Emotion detection (timed)
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

                        # Save face region for analysis
                        face_img = frame[face_area["y"]:face_area["y"] + face_area["h"],
                                   face_area["x"]:face_area["x"] + face_area["w"]]
                        cv2.imwrite(self.TEMP_IMG_PATH, face_img)

                        # 68-point feature detection
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces_dlib = detector_dlib(gray)
                        current_micro = "neutral"

                        for face in faces_dlib:
                            landmarks = predictor(gray, face)
                            features = extract_features(landmarks, face)
                            micro_expression = classify_micro_expression(features)
                            # Convert to English
                            micro_expression = {
                                "自然": "neutral",
                                "开心": "happy",
                                "愤怒": "angry",
                                "悲伤": "sad",
                                "惊讶": "surprise",
                                "厌恶": "disgust",
                                "恐惧": "fear"
                            }.get(micro_expression, micro_expression)
                            micro_expression_window.append(micro_expression)
                            current_micro = max(set(micro_expression_window), key=micro_expression_window.count)

                        # Main emotion analysis
                        if not self.demographics_initialized:
                            results = DeepFace.analyze(
                                img_path=self.TEMP_IMG_PATH,
                                actions=["emotion", "age", "gender", "race"],
                                detector_backend="skip",
                                enforce_detection=False,
                                silent=True
                            )
                            if results:
                                r = results[0]
                                self.demographics["age"] = int(r["age"])
                                self.demographics["gender"] = r["dominant_gender"]
                                self.demographics["gender_confidence"] = r["gender"][r["dominant_gender"]]
                                self.demographics["race"] = r["dominant_race"]
                                self.demographics["race_confidence"] = r["race"][r["dominant_race"]]
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

                        # Combine results using English rules
                        refined_emotion = get_refined_emotion_en(current_emotion, current_micro)
                        confidence = get_confidence_en(current_emotion, current_micro)

                        # Update results
                        with self.lock:
                            self.latest_result = {
                                "emotion": current_emotion,
                                "emotion_index": self.EMOTION_CLASSES.index(current_emotion.capitalize()) if current_emotion.capitalize() in self.EMOTION_CLASSES else 6,
                                "probability": 1.0,
                                "all_probabilities": {emo: 1.0 if emo.lower() == current_emotion.lower() else 0.0 for emo in self.EMOTION_CLASSES},
                                "timestamp": time.time(),
                                "heart_rate": int(self.locked_hr_value) if self.locked_hr_value is not None else None,
                                "demographics": self.demographics.copy(),
                                "micro_expression": current_micro,
                                "refined_emotion": refined_emotion,
                                "confidence": confidence
                            }
                            if self.callback:
                                self.callback(self.latest_result)

                        last_detection_time = current_time

                    if os.path.exists(self.TEMP_IMG_PATH):
                        os.remove(self.TEMP_IMG_PATH)

                except Exception as e:
                    print(f"Detection failed: {e}")
                    if os.path.exists(self.TEMP_IMG_PATH):
                        os.remove(self.TEMP_IMG_PATH)

            # 5. Display video
            if show_video:
                cv2.imshow('Emotion & Heart Rate Detection', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    break

        # Cleanup
        if self.cap is not None:
            self.cap.release()
        if show_video:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    def print_result(result):
        print(f"\nDetection time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Emotion: {result['emotion']}")
        print(f"Micro Expression: {result.get('micro_expression', 'N/A')}")
        print(f"Refined Emotion: {result.get('refined_emotion', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        if result.get('heart_rate') is not None:
            print(f"Heart Rate: {result['heart_rate']:.1f} BPM")
        else:
            print("Heart Rate: Calculating...")
        print("Probabilities for all categories:")
        for emo, prob in result['all_probabilities'].items():
            print(f"  {emo}: {prob * 100:.2f}%")
        if result.get('demographics'):
            demo = result['demographics']
            print(f"Demographics - Age: {demo['age']}, Gender: {demo['gender']}, Race: {demo['race']}")


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