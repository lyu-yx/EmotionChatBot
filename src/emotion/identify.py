import cv2
import time
import os
import threading
from typing import Dict, Any, Optional, Callable
from PIL import ImageFont, ImageDraw, Image
import collections
import dlib
from deepface import DeepFace
import numpy as np
import mediapipe as mp
import math
# 多帧投票窗口大小
VOTE_WINDOW = 5
emotion_window = collections.deque(maxlen=VOTE_WINDOW)

# 情感偏置权重（用于调整某些情感的敏感度）
bias_weights = {
    "happy": 1.5,
    "neutral": 1.5,
    "sad": 0.1,
    "angry": 0.1,
    "fear": 0.7,
    "disgust": 1.0,
    "surprise": 1.6
}


def cv2_putText_cn(img, text, position, font_path="simhei.ttf", font_size=32, color=(0, 255, 0)):
    """在OpenCV图像上绘制中文文本"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        raise RuntimeError(f"字体加载失败: {e}")
    draw.text(position, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class EmotionDetectorCamera:
    # 情感类别定义（英文和中文）
    EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    EMOTION_CLASSES_ZH = ['愤怒', '厌恶', '恐惧', '开心', '悲伤', '惊讶', '平静']
    MICRO_EXPRESSION_CLASSES = ['content', 'melancholy', 'irritated', 'curious', 'apprehensive', 'unimpressed',
                                'moved', 'frustrated', 'delighted', 'amused', 'excited', 'smile', 'relieved',
                                'resentful', 'touched', 'lonely', 'disappointed', 'passionate', 'gloomy',
                                'outraged', 'threatening', 'hostile', 'amazed', 'shocked', 'alarmed',
                                'panicked', 'appalled', 'thrilled', 'anxious', 'defensive', 'nervous',
                                'repulsed', 'sarcastic', 'contempt', 'loathing', 'uncomfortable', 'revolted',
                                'phobic', 'confident', 'suspicious', 'confused', 'playful', 'jealous',
                                'embarrassed', 'expectant', 'regretful', 'charming', 'determined', 'desirous',
                                'perplexed', 'eager', 'excited', 'nervous', 'alert', 'composed', 'pensive',
                                'intoxicated', 'puzzled']

    def __init__(self,
                 detection_interval: float = 0.5,
                 use_chinese: bool = False,
                 callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        初始化情感检测器

        参数:
            detection_interval: 检测间隔时间(秒)
            use_chinese: 是否使用中文显示
            callback: 检测结果回调函数
        """
        self.calibration_start_time = None
        self.is_calibrating = False
        self.calibration_complete = False
        self.neutral_features = []
        self.neutral_landmark_distances = None
        self.in_micro_expression = False
        self.last_macro_emotion_time = time.time()
        # 新增微表情投票窗口
        self.micro_expression_window = collections.deque(maxlen=5)  # 5帧窗口
        self.current_stable_micro = None  # 当前稳定的微表情
        self.calibration_duration = 3  # 3秒标定时间
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
            "timestamp": time.time(),
            "micro_expression": None
        }
        # 分心检测基线
        self.eye_relative_baseline = None  # {'left': (x, y), 'right': (x, y)}
        self.distracted = False
        self.distract_start_time = None  # 分心计时起点
        self.need_recalibration = False  # 标记是否需要重新标定
        self.recalibration_start_time = None  # neutral重新计时

        # 初始化OpenCV人脸检测器
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # 初始化dlib面部关键点检测器
        self.landmark_detector = dlib.shape_predictor("/home/liugezhi/下载/shape_predictor_68_face_landmarks.dat")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.LEFT_EYE_INDICES = [33, 133, 159, 145, 153, 144, 160, 158]
        self.RIGHT_EYE_INDICES = [362, 263, 386, 374, 380, 373, 387, 385]
        self.LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

        # 疲劳检测相关变量
        self.eye_counter = 0
        self.yawn_counter = 0
        self.eye_closed = False
        self.yawn_detected = False
        self.fatigue_level = 0
        # 临时文件路径
        self.TEMP_IMG_PATH = "temp_frame.jpg"
        self.last_valid_face_rect = None

        # rPPG心率检测相关变量
        self.rppg_green_buffer = []  # 存储绿色通道均值
        self.rppg_time_buffer = []   # 存储时间戳
        self.rppg_buffer_size = 300  # 约10秒（30fps）
        self.rppg_last_bpm = None
        self.rppg_last_update = 0
        self.rppg_update_interval = 2.0  # 每2秒更新一次心率



    def show_text(self, frame, text, position=(50, 50), color=(0, 255, 0), size=1.0):
        """在帧上显示文本(支持中文)"""
        if self.emotion_classes == self.EMOTION_CLASSES_ZH:
            frame = cv2_putText_cn(frame, text, position, font_size=int(32 * size), color=color)
        else:
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)
        return frame

    def start(self, camera_id: int = 0, show_video: bool = False):
        """启动检测线程"""
        if self.is_running:
            print("情感检测已在运行中")
            return False

        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                print(f"无法打开摄像头 ID: {camera_id}")
                return False

            self.is_running = True
            self.detection_thread = threading.Thread(
                target=self._detection_loop,
                args=(show_video,),
                daemon=True
            )
            self.detection_thread.start()
            print(f"情感检测已成功启动，使用摄像头 ID: {camera_id}")
            return True

        except Exception as e:
            print(f"启动情感检测失败: {str(e)}")
            self.is_running = False
            if self.cap is not None:
                self.cap.release()
            return False

    def stop(self):
        """停止检测"""
        if not self.is_running:
            return

        self.is_running = False
        if self.detection_thread is not None and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("情感检测已停止")

    def get_latest_emotion(self) -> Dict[str, Any]:
        """获取最新检测结果"""
        with self.lock:
            return self.latest_result.copy()


    




    def _extract_features(self, landmarks, face_rect):
        """提取面部特征"""
        face_width = face_rect.right() - face_rect.left()
        face_height = face_rect.bottom() - face_rect.top()

        # 嘴巴特征
        mouth_width = (landmarks.part(54).x - landmarks.part(48).x) / face_width
        mouth_higth = (landmarks.part(59).x - landmarks.part(48).x) / face_width
        mouth_core_width = (landmarks.part(59).x - landmarks.part(48).x) / face_width
        mouth_core_hight = (landmarks.part(60).y - landmarks.part(48).y) / face_width
        eye_core_hight = (landmarks.part(42).y - landmarks.part(22).y) / face_width

        # 眉毛特征
        brow_sum = 0
        frown_sum = 0
        line_brow_x = []
        line_brow_y = []

        for j in range(17, 21):
            brow_sum += (landmarks.part(j).y - face_rect.top()) + (landmarks.part(j + 5).y - face_rect.top())
            frown_sum += landmarks.part(j + 5).x - landmarks.part(j).x
            line_brow_x.append(landmarks.part(j).x)
            line_brow_y.append(landmarks.part(j).y)

        tempx = np.array(line_brow_x)
        tempy = np.array(line_brow_y)
        brow_k = -round(np.polyfit(tempx, tempy, 1)[0], 3) if len(tempx) > 0 else 0
        brow_hight = (landmarks.part(43).y - landmarks.part(23).y) / face_width
        brow_width = (frown_sum / 5) / face_width

        # 眼睛特征
        eye_sum = (landmarks.part(41).y - landmarks.part(37).y +
                   landmarks.part(40).y - landmarks.part(38).y +
                   landmarks.part(47).y - landmarks.part(43).y +
                   landmarks.part(46).y - landmarks.part(44).y)
        eye_hight = (eye_sum / 4) / face_width

        # 鼻子特征
        nose_wrinkling = (landmarks.part(31).y - landmarks.part(27).y) / face_height

        left_eye_center = ((landmarks.part(36).x + landmarks.part(39).x) / 2,
                           (landmarks.part(36).y + landmarks.part(39).y) / 2)
        right_eye_center = ((landmarks.part(42).x + landmarks.part(45).x) / 2,
                            (landmarks.part(42).y + landmarks.part(45).y) / 2)

        # 计算左右嘴角
        left_mouth = (landmarks.part(48).x, landmarks.part(48).y)
        right_mouth = (landmarks.part(54).x, landmarks.part(54).y)

        # 计算左右眉毛中心
        left_brow_center = ((landmarks.part(17).x + landmarks.part(21).x) / 2,
                            (landmarks.part(17).y + landmarks.part(21).y) / 2)
        right_brow_center = ((landmarks.part(22).x + landmarks.part(26).x) / 2,
                             (landmarks.part(22).y + landmarks.part(26).y) / 2)

        # 计算对称性误差
        face_width = face_rect.right() - face_rect.left()
        face_height = face_rect.bottom() - face_rect.top()

        # 水平对称性 (y坐标差异)
        eye_y_diff = abs(left_eye_center[1] - right_eye_center[1]) / face_height
        brow_y_diff = abs(left_brow_center[1] - right_brow_center[1]) / face_height
        mouth_y_diff = abs(left_mouth[1] - right_mouth[1]) / face_height

        # 垂直对称性 (x坐标相对于面部中心的差异)
        face_center_x = face_rect.left() + face_width / 2
        left_eye_x_diff = abs((left_eye_center[0] - face_center_x) - (face_center_x - right_eye_center[0])) / face_width
        left_brow_x_diff = abs((left_brow_center[0] - face_center_x) - (face_center_x - right_brow_center[0])) / face_width
        left_mouth_x_diff = abs((left_mouth[0] - face_center_x) - (face_center_x - right_mouth[0])) / face_width

        return {
            'eye_y_diff': eye_y_diff,
            'brow_y_diff': brow_y_diff,
            'mouth_y_diff': mouth_y_diff,
            'left_eye_x_diff': left_eye_x_diff,
            'left_brow_x_diff': left_brow_x_diff,
            'left_mouth_x_diff': left_mouth_x_diff,
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

    def _detect_micro_expression(self, current_landmark_distances):
        """检测微表情"""
        if self.neutral_landmark_distances is None or current_landmark_distances is None:
            return None

        # 计算特征变化
        changes = {}
        for name, current_val in current_landmark_distances.items():
            neutral_val = self.neutral_landmark_distances.get(name, 0)
            if neutral_val != 0:
                changes[name] = (current_val - neutral_val) / neutral_val * 100  # 百分比变化
        micro_expression = None
        # 根据特征变化判断微表情

        if changes.get('brow_hight', 0) < -15:  # 眉毛下垂超过15%
            primary = "angry"
        elif abs(changes.get('eye_hight', 0)) > 30:  # 眼睛变化超过30%
            primary = "fear"
        elif changes.get('nose_wrinkling', 0) > 5:  # 鼻子皱起超过5%
            primary = "disgust"
        elif changes.get('brow_hight', 0) > 8.5:  # 眉毛上扬超过8.5%
            primary = "surprise"
        elif changes.get('mouth_higth', 0) > 5:  # 嘴角上扬超过5%
            primary = "happy"
        elif changes.get('mouth_core_hight', 0)  >30:  # 嘴角下垂超过-8%
            primary = "sad"
        else:
            primary = "neutral"

        # 检测次要表情
        secondary = "neutral"
        if abs(changes.get('eye_hight', 0)) > 40 and primary != "fear":
            secondary = "fear"
        elif changes.get('nose_wrinkling', 0) > 4 and primary != "disgust":
            secondary = "disgust"
        elif changes.get('brow_hight', 0) > 10 and primary != "surprise":
            secondary = "surprise"
        elif changes.get('mouth_higth', 0) > 7 and primary != "happy":
            secondary = "happy"
        elif changes.get('mouth_core_hight', 0) >30 and primary != "sad":
            secondary = "sad"
        elif changes.get('brow_hight', 0) < -20 and primary != "angry":
            secondary = "angry"

        # 组合表情判断 (按照优先级排序)
        if primary == "happy" and secondary == "sad":
            micro_expression = "moved"
        elif primary == "happy" and secondary == "angry":
            micro_expression = "frustrated"
        elif primary == "happy" and secondary == "surprise":
            micro_expression = "delighted"
        elif primary == "happy" and secondary == "disgust":
            micro_expression = "amused"
        elif primary == "happy" and secondary == "fear":
            micro_expression = "excited"
        elif primary == "happy" and secondary == "neutral":
            micro_expression = "smile"
        elif primary == "sad" and secondary == "happy":
            micro_expression = "relieved"
        elif primary == "sad" and secondary == "angry":
            micro_expression = "resentful"
        elif primary == "sad" and secondary == "surprise":
            micro_expression = "touched"
        elif primary == "sad" and secondary == "neutral":
            micro_expression = "sad"
        elif primary == "sad" and secondary == "fear":
            micro_expression = "lonely"
        elif primary == "sad" and secondary == "disgust":
            micro_expression = "disappointed"
        elif primary == "angry" and secondary == "happy":
            micro_expression = "passionate"
        elif primary == "angry" and secondary == "sad":
            micro_expression = "gloomy"
        elif primary == "angry" and secondary == "neutral":
            micro_expression = "angry"
        elif primary == "angry" and secondary == "surprise":
            micro_expression = "outraged"
        elif primary == "angry" and secondary == "fear":
            micro_expression = "threatening"
        elif primary == "angry" and secondary == "disgust":
            micro_expression = "hostile"
        elif primary == "surprise" and secondary == "happy":
            micro_expression = "amazed"
        elif primary == "surprise" and secondary == "sad":
            micro_expression = "shocked"
        elif primary == "surprise" and secondary == "angry":
            micro_expression = "alarmed"
        elif primary == "surprise" and secondary == "neutral":
            micro_expression = "curious"
        elif primary == "surprise" and secondary == "fear":
            micro_expression = "panicked"
        elif primary == "surprise" and secondary == "disgust":
            micro_expression = "appalled"
        elif primary == "fear" and secondary == "happy":
            micro_expression = "thrilled"
        elif primary == "fear" and secondary == "sad":
            micro_expression = "anxious"
        elif primary == "fear" and secondary == "angry":
            micro_expression = "defensive"
        elif primary == "fear" and secondary == "neutral":
            micro_expression = "nervous"
        elif primary == "fear" and secondary == "surprise":
            micro_expression = "terrified"
        elif primary == "fear" and secondary == "disgust":
            micro_expression = "repulsed"
        elif primary == "disgust" and secondary == "happy":
            micro_expression = "sarcastic"
        elif primary == "disgust" and secondary == "sad":
            micro_expression = "contempt"
        elif primary == "disgust" and secondary == "angry":
            micro_expression = "loathing"
        elif primary == "disgust" and secondary == "neutral":
            micro_expression = "uncomfortable"
        elif primary == "disgust" and secondary == "surprise":
            micro_expression = "revolted"
        elif primary == "disgust" and secondary == "fear":
            micro_expression = "phobic"
        elif primary == "neutral" and secondary == "happy":
            micro_expression = "content"
        elif primary == "neutral" and secondary == "sad":
            micro_expression = "melancholy"
        elif primary == "neutral" and secondary == "angry":
            micro_expression = "irritated"
        elif primary == "neutral" and secondary == "surprise":
            micro_expression = "curious"
        elif primary == "neutral" and secondary == "fear":
            micro_expression = "apprehensive"
        elif primary == "neutral" and secondary == "disgust":
            micro_expression = "unimpressed"
        else:
            micro_expression = primary  # 如果没有匹配的组合，返回主表情

        if (changes.get('eye_hight', 0) > 20) and (changes.get('brow_hight', 0) > 10):
            micro_expression = "curious"
        # 自信 - 眼睛开合增加15%-25%，口角上扬3%-5%，眉毛变化不大
        elif (15 <= changes.get('eye_hight', 0) <= 25) and (3 <= changes.get('mouth_higth', 0) <= 5):
            micro_expression = "confident"
        # 怀疑 - 眉毛上扬>15%，口角下垂>=-3%，眼睛开合变化不大
        elif (changes.get('brow_hight', 0) > 15) and (changes.get('mouth_higth', 0) <= -3):
            micro_expression = "suspicious"
        elif (changes.get('brow_hight', 0) < -10) and (changes.get('mouth_hight', 0) > 2) and (abs(changes.get('eye_hight', 0)) < 10):
            micro_expression = "confused"
        # 调皮
        elif (changes.get('eye_hight', 0) <= 10) and (changes.get('mouth_hight', 0) > 5) and (changes.get('brow_hight', 0) <= 5) and (abs(changes.get('mouth_edge_distance', 0)) < 2):
            micro_expression = "playful"
        # 嫉妒
        elif (changes.get('eye_hight', 0) < -10) and (changes.get('mouth_hight', 0) < -5) and (changes.get('brow_hight', 0) > 8) and (changes.get('eyelid_hight', 0) < -5):
            micro_expression = "jealous"
        # 尴尬
        elif (abs(changes.get('eye_hight', 0)) < 10) and (abs(changes.get('mouth_hight', 0)) < 5) and (changes.get('brow_hight', 0) < -15) and (changes.get('nose_bridge_hight', 0) > 5):
            micro_expression = "embarrassed"
        # 期待
        elif (changes.get('brow_hight', 0) > 15) and (abs(changes.get('mouth_hight', 0)) < 5) and (changes.get('eye_hight', 0) > 20) and (changes.get('eyelid_hight', 0) < -5):
            micro_expression = "expectant"
        # 后悔
        elif (changes.get('brow_hight', 0) < -15) and (changes.get('mouth_hight', 0) < -5) and (changes.get('eye_hight', 0) < -10) and (changes.get('brow_inner_distance', 0) > 5):
            micro_expression = "regretful"
        # 魅力
        elif (changes.get('eye_hight', 0) <= 15) and (3 <= changes.get('mouth_hight', 0) <= 5) and (changes.get('brow_hight', 0) <= 8) and (changes.get('eyelid_hight', 0) <= 5):
            micro_expression = "charming"
        # 坚定
        elif (changes.get('brow_hight', 0) < -10) and (changes.get('mouth_hight', 0) <= 3) and (changes.get('eye_hight', 0) <= 5) and (changes.get('eyelid_hight', 0) < -5):
            micro_expression = "determined"
        # 渴望
        elif (changes.get('eye_hight', 0) > 25) and (changes.get('mouth_hight', 0) > 5) and (changes.get('brow_hight', 0) <= 10) and (changes.get('eyelid_hight', 0) < -5):
            micro_expression = "desirous"
        # 迷惑
        elif (changes.get('brow_hight', 0) < -15) and (changes.get('mouth_open', 0) > 5) and (abs(changes.get('eye_hight', 0)) < 10):
            micro_expression = "perplexed"
        # 急切
        elif (10 <= changes.get('brow_hight', 0) <= 15) and (changes.get('mouth_hight', 0) > 5) and (changes.get('eye_hight', 0) > 30) and (changes.get('eyelid_hight', 0) < -5):
            micro_expression = "eager"
        # 激动
        elif (changes.get('eye_hight', 0) > 40) and (changes.get('mouth_hight', 0) > 10) and (changes.get('brow_hight', 0) > 15) and (changes.get('eyelid_hight', 0) <= 5):
            micro_expression = "excited"
        # 紧张
        elif (changes.get('eye_hight', 0) <= 25) and (changes.get('mouth_hight', 0) < -5) and (changes.get('brow_hight', 0) > 8) and (changes.get('eyelid_hight', 0) < -10):
            micro_expression = "nervous"
        # 警觉
        elif (changes.get('eye_hight', 0) > 50) and (changes.get('brow_hight', 0) > 20) and (changes.get('mouth_open', 0) > 10) and (changes.get('eyelid_hight', 0) <= 10):
            micro_expression = "alert"
        # 从容
        elif (changes.get('brow_hight', 0) < -10) and (abs(changes.get('mouth_hight', 0)) < 3) and (abs(changes.get('eye_hight', 0)) < 10) and (changes.get('eyelid_hight', 0) < -5):
            micro_expression = "composed"
        # 沉思
        elif (changes.get('brow_hight', 0) < -15) and (abs(changes.get('mouth_hight', 0)) < 5) and (abs(changes.get('eye_hight', 0)) < 10) and (changes.get('eyelid_hight', 0) < -10):
            micro_expression = "pensive"
        # 陶醉
        elif (changes.get('eye_hight', 0) > 30) and (changes.get('mouth_hight', 0) > 15) and (changes.get('brow_hight', 0) > 10) and (changes.get('eyelid_hight', 0) < -10):
            micro_expression = "intoxicated"
        # 迷惑
        elif (changes.get('brow_hight', 0) > 8) and (abs(changes.get('mouth_hight', 0)) < 3) and (abs(changes.get('eye_hight', 0)) < 10) and (changes.get('eyelid_hight', 0) < -5):
            micro_expression = "puzzled"
        return micro_expression

    def _detection_loop(self, show_video: bool):
        """检测主循环（整合68点关键点绘制）"""
        last_detection_time = 0
        current_emotion = "neutral"
        current_micro_expression = None
        last_calibration_data = None  # 保存上次成功的标定数据

        # 虹膜检测相关变量
        left_iris_ellipse = None
        right_iris_ellipse = None
        left_eye_ellipse = None
        right_eye_ellipse = None

        # 疲劳检测相关变量
        eye_ratio_threshold = 0.1  # 眼睛横纵比阈值，小于此值认为闭眼
        mouth_ratio_threshold = 0.9  # 嘴巴横纵比阈值，大于此值认为打哈欠
        eye_closed_frames = 0  # 连续闭眼帧数
        yawn_frames = 0  # 连续打哈欠帧数
        fatigue_warning = False  # 疲劳警告状态

        # 显示参数配置
        font_scale = 0.6  # 字体大小
        text_color = (0, 0, 255)  # 文字颜色(红色)
        text_thickness = 1  # 文字粗细
        line_height = 25  # 行间距
        text_x = 10  # 起始x坐标

        while self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取视频帧")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420) if frame.shape[-1] != 3 else frame
            current_time = time.time()
            display_frame = frame.copy()

            # 在画面底部显示状态信息
            y_pos = display_frame.shape[0] - 50
            # 只在自然表情时显示微表情作为主要情感
            if current_emotion == "neutral" and current_micro_expression:
                final_emotion_display = current_micro_expression
                cv2.putText(display_frame, f"Emotion: {final_emotion_display}",
                            (text_x, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)
                # 显示基础情感作为参考
                cv2.putText(display_frame, f"Base: {current_emotion}",
                            (text_x, y_pos - line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thickness)
            else:
                # 非自然表情时，直接显示主要情感
                cv2.putText(display_frame, f"Emotion: {current_emotion}",
                            (text_x, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)

            # 使用OpenCV检测人脸
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)
            )

            if len(faces) > 0:
                # 选择最大的人脸
                main_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = main_face
                self.last_valid_face_rect = (x, y, w, h)

                # 裁剪人脸区域
                face_img = frame[y:y + h, x:x + w]

                # rPPG心率检测：提取绿色通道均值
                face_roi = frame[y:y + h, x:x + w]
                if face_roi.size > 0:
                    green_mean = np.mean(face_roi[:, :, 1])
                    self.rppg_green_buffer.append(green_mean)
                    self.rppg_time_buffer.append(current_time)
                    # 保持缓冲区长度
                    if len(self.rppg_green_buffer) > self.rppg_buffer_size:
                        self.rppg_green_buffer = self.rppg_green_buffer[-self.rppg_buffer_size:]
                        self.rppg_time_buffer = self.rppg_time_buffer[-self.rppg_buffer_size:]

                # rPPG心率估算
                if len(self.rppg_green_buffer) >= int(self.rppg_buffer_size * 0.8):
                    if current_time - self.rppg_last_update > self.rppg_update_interval:
                        # 去均值
                        signal = np.array(self.rppg_green_buffer)
                        signal = signal - np.mean(signal)
                        # 采样率
                        duration = self.rppg_time_buffer[-1] - self.rppg_time_buffer[0]
                        if duration > 0:
                            fps = len(self.rppg_time_buffer) / duration
                            # FFT
                            freqs = np.fft.rfftfreq(len(signal), d=1.0/fps)
                            fft = np.abs(np.fft.rfft(signal))
                            # 心率范围（0.8Hz-3Hz, 48-180bpm）
                            mask = (freqs >= 0.8) & (freqs <= 3.0)
                            if np.any(mask):
                                peak_freq = freqs[mask][np.argmax(fft[mask])]
                                bpm = int(peak_freq * 60)
                                self.rppg_last_bpm = bpm
                                self.rppg_last_update = current_time

                # 持续进行关键点检测和绘制
                if self.landmark_detector:
                    # 转换为灰度图
                    gray_roi = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    dlib_rect = dlib.rectangle(0, 0, gray_roi.shape[1], gray_roi.shape[0])

                    # 检测关键点
                    landmarks = self.landmark_detector(gray_roi, dlib_rect)

                    # 绘制所有68个关键点
                    for i in range(68):
                        point = (landmarks.part(i).x + x, landmarks.part(i).y + y)
                        cv2.circle(display_frame, point, 2, (0, 255, 255), -1)

                    # 绘制关键区域连线
                    # 下巴线（0-16）
                    for i in range(16):
                        cv2.line(display_frame,
                                 (landmarks.part(i).x + x, landmarks.part(i).y + y),
                                 (landmarks.part(i + 1).x + x, landmarks.part(i + 1).y + y),
                                 (255, 0, 0), 1)

                    # 左眉毛（17-21）
                    for i in range(17, 21):
                        cv2.line(display_frame,
                                 (landmarks.part(i).x + x, landmarks.part(i).y + y),
                                 (landmarks.part(i + 1).x + x, landmarks.part(i + 1).y + y),
                                 (0, 255, 0), 1)

                    # 右眉毛（22-26）
                    for i in range(22, 26):
                        cv2.line(display_frame,
                                 (landmarks.part(i).x + x, landmarks.part(i).y + y),
                                 (landmarks.part(i + 1).x + x, landmarks.part(i + 1).y + y),
                                 (0, 255, 0), 1)

                    # 鼻子（27-35）
                    for i in range(27, 35):
                        cv2.line(display_frame,
                                 (landmarks.part(i).x + x, landmarks.part(i).y + y),
                                 (landmarks.part(i + 1).x + x, landmarks.part(i + 1).y + y),
                                 (0, 0, 255), 1)

                    # 外唇（48-59）
                    for i in range(48, 60):
                        start = (landmarks.part(i).x + x, landmarks.part(i).y + y)
                        end = (landmarks.part(i + 1).x + x, landmarks.part(i + 1).y + y) if i < 59 else (landmarks.part(48).x + x, landmarks.part(48).y + y)
                        cv2.line(display_frame, start, end, (0, 255, 255), 1)

                    # 内唇（60-67）
                    for i in range(60, 68):
                        start = (landmarks.part(i).x + x, landmarks.part(i).y + y)
                        end = (landmarks.part(i + 1).x + x, landmarks.part(i + 1).y + y) if i < 67 else (landmarks.part(60).x + x, landmarks.part(60).y + y)
                        cv2.line(display_frame, start, end, (255, 0, 255), 1)

                    # 疲劳检测：计算眼睛和嘴巴的横纵比
                    # 左眼横纵比 (36-39, 37-38)
                    left_eye_width = abs(landmarks.part(39).x - landmarks.part(36).x)
                    left_eye_height = abs(landmarks.part(37).y - landmarks.part(41).y)
                    left_eye_ratio = left_eye_height / left_eye_width if left_eye_width > 0 else 0

                    # 右眼横纵比 (42-45, 43-44)
                    right_eye_width = abs(landmarks.part(45).x - landmarks.part(42).x)
                    right_eye_height = abs(landmarks.part(43).y - landmarks.part(47).y)
                    right_eye_ratio = right_eye_height / right_eye_width if right_eye_width > 0 else 0

                    # 嘴巴横纵比 (48-54, 51-57)
                    mouth_width = abs(landmarks.part(54).x - landmarks.part(48).x)
                    mouth_height = abs(landmarks.part(51).y - landmarks.part(57).y)
                    mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0

                    # 判断闭眼
                    eyes_closed = (left_eye_ratio < eye_ratio_threshold) or (right_eye_ratio < eye_ratio_threshold)
                    if eyes_closed:
                        eye_closed_frames += 1
                    else:
                        eye_closed_frames = 0

                    # 判断打哈欠
                    is_yawn = mouth_ratio > mouth_ratio_threshold
                    if is_yawn:
                        yawn_frames += 1
                    else:
                        yawn_frames = 0

                    # 疲劳警告逻辑
                    if eye_closed_frames >= 10 or yawn_frames >= 5:  # 连续闭眼10帧或打哈欠5帧
                        fatigue_warning = True
                    else:
                        fatigue_warning = False

                    # 自然状态标定逻辑
                    if not self.calibration_complete:
                        if not self.is_calibrating and current_emotion == "neutral":
                            print("请保持自然表情3秒进行标定...")
                            self.is_calibrating = True
                            self.calibration_start_time = current_time
                            self.neutral_features = []  # 重置标定数据

                        if self.is_calibrating:
                            if current_emotion != "neutral":
                                print("检测到非中性表情，标定中断！")
                                self.is_calibrating = False
                            elif current_time - self.calibration_start_time < self.calibration_duration:
                                # 收集中性状态特征
                                features = self._extract_features(landmarks, dlib_rect)
                                self.neutral_features.append(features)
                                # 显示标定倒计时
                                remaining = int(self.calibration_duration - (current_time - self.calibration_start_time))
                                cv2.putText(display_frame, f"Calibrating... {remaining}s",
                                            (x, y - 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            else:
                                # 标定完成，计算中性状态特征平均值
                                self.is_calibrating = False
                                self.calibration_complete = True

                                # 计算各特征的平均值
                                avg_features = {}
                                for key in self.neutral_features[0].keys():
                                    avg_features[key] = sum(f[key] for f in self.neutral_features) / len(self.neutral_features)

                                self.neutral_landmark_distances = avg_features
                                last_calibration_data = avg_features  # 保存本次标定数据
                                print("标定完成！中性状态特征已保存。")
                    else:
                        # 标定完成后，检测是否需要重新标定
                        if self.calibration_complete:
                            if current_emotion != "neutral":
                                self.need_recalibration = True
                                self.recalibration_start_time = None
                            elif self.need_recalibration and current_emotion == "neutral":
                                if self.recalibration_start_time is None:
                                    self.recalibration_start_time = current_time
                                elif current_time - self.recalibration_start_time >= self.calibration_duration:
                                    # 重新标定
                                    print("检测到回到自然表情，自动重新标定...")
                                    self.calibration_complete = False
                                    self.is_calibrating = True
                                    self.need_recalibration = False
                                    self.calibration_start_time = current_time
                                    self.neutral_features = []
                            else:
                                self.recalibration_start_time = None
                        # 标定完成后进行微表情检测
                        # 不再因表情变化重置标定，分心检测始终有效
                        # if current_emotion != "neutral":
                        #     self.calibration_complete = False
                        #     self.is_calibrating = False
                        #     print("检测到非中性表情，已重置标定状态！")
                        # else:
                        #     # 标定完成后进行微表情检测
                        #     current_features = self._extract_features(landmarks, dlib_rect)
                        #     detected_micro = self._detect_micro_expression(current_features)
                        #
                        #     if detected_micro is not None:
                        #         self.micro_expression_window.append(detected_micro)
                        #
                        #         # 统计窗口中最频繁的微表情
                        #         if len(self.micro_expression_window) > 0:
                        #             micro_counts = collections.Counter(self.micro_expression_window)
                        #             most_common = micro_counts.most_common(1)[0]
                        #             if most_common[1] >= 3:  # 至少出现3次才认为是稳定的微表情
                        #                 self.current_stable_micro = most_common[0]
                        #             else:
                        #                 self.current_stable_micro = None
                        #
                        #     current_micro_expression = self.current_stable_micro
                        # 微表情检测逻辑 - 只在自然表情时检测
                        if current_emotion == "neutral":
                            current_features = self._extract_features(landmarks, dlib_rect)
                            detected_micro = self._detect_micro_expression(current_features)
                            if detected_micro is not None:
                                self.micro_expression_window.append(detected_micro)
                                if len(self.micro_expression_window) > 0:
                                    micro_counts = collections.Counter(self.micro_expression_window)
                                    most_common = micro_counts.most_common(1)[0]
                                    if most_common[1] >= 3:
                                        self.current_stable_micro = most_common[0]
                                    else:
                                        self.current_stable_micro = None
                            current_micro_expression = self.current_stable_micro
                        else:
                            # 非自然表情时，清空微表情检测
                            self.micro_expression_window.clear()
                            self.current_stable_micro = None
                            current_micro_expression = None

            # 定时执行情感检测
            if current_time - last_detection_time >= self.detection_interval:
                try:
                    if len(faces) > 0:
                        # 分析情感（跳过检测步骤）
                        results = DeepFace.analyze(
                            img_path=face_img,
                            actions=["emotion"],
                            detector_backend="skip",
                            enforce_detection=False,
                            silent=True
                        )

                        if results:
                            # 应用情感偏置权重
                            raw_emotions = results[0]["emotion"]
                            biased_emotions = {emo: raw_emotions[emo] * bias_weights.get(emo, 1.0) for emo in raw_emotions}
                            emotion_window.append(biased_emotions)

                            # 计算多帧平均情感分数
                            combined_scores = {}
                            for e in emotion_window:
                                for emo, score in e.items():
                                    combined_scores[emo] = combined_scores.get(emo, 0) + score
                            for emo in combined_scores:
                                combined_scores[emo] /= len(emotion_window)

                            # 确定当前主要情感
                            current_emotion = max(combined_scores, key=combined_scores.get)

                        # 更新最新结果 - 只在自然表情时使用微表情作为最终结果
                        if current_emotion == "neutral" and current_micro_expression:
                            final_emotion = current_micro_expression
                        else:
                            final_emotion = current_emotion
                        
                        with self.lock:
                            self.latest_result = {
                                "emotion": final_emotion,
                                "emotion_index": self.EMOTION_CLASSES.index(current_emotion) if current_emotion in self.EMOTION_CLASSES else 6,
                                "probability": 1.0,
                                "all_probabilities": {emo: 1.0 if emo.lower() == final_emotion.lower() else 0.0 for emo in self.emotion_classes},
                                "timestamp": time.time(),
                                "micro_expression": current_micro_expression
                            }
                            if self.callback:
                                self.callback(self.latest_result)

                    last_detection_time = current_time

                except Exception as e:
                    print(f"检测失败: {e}")

            # 绘制人脸矩形框（保留原有逻辑）
            if self.last_valid_face_rect:
                x, y, w_rect, h_rect = self.last_valid_face_rect
                cv2.rectangle(display_frame, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 2)
                text_y_start = 30
                line_height = 30
                font_scale = 0.7
                thickness = 2
                # 显示情感标签
                if current_emotion == "neutral" and current_micro_expression:
                    # 自然表情时显示微表情作为主要标签
                    emotion_text = f"{current_micro_expression}"
                    text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_x_pos = x + (w_rect - text_size[0]) // 2
                    cv2.putText(display_frame, emotion_text,
                                (text_x_pos, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # 显示基础情感作为参考
                    base_text = f"Base: {current_emotion}"
                    text_size = cv2.getTextSize(base_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_x_pos = x + (w_rect - text_size[0]) // 2
                    cv2.putText(display_frame, base_text,
                                (text_x_pos, y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    # 非自然表情时，直接显示主要情感
                    emotion_text = f"{current_emotion}"
                    text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_x_pos = x + (w_rect - text_size[0]) // 2
                    cv2.putText(display_frame, emotion_text,
                                (text_x_pos, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示疲劳检测标签
                if fatigue_warning:
                    fatigue_text = "FATIGUE!"
                    text_size = cv2.getTextSize(fatigue_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    text_x_pos = x + (w_rect - text_size[0]) // 2
                    cv2.putText(display_frame, fatigue_text,
                                (text_x_pos, y - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 显示心率
            if self.rppg_last_bpm is not None:
                cv2.putText(display_frame, f"Heart Rate: {self.rppg_last_bpm} bpm", (text_x, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            left_iris_ellipse = None
            right_iris_ellipse = None
            left_eye_ellipse = None
            right_eye_ellipse = None
            left_relative = None
            right_relative = None

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h_img, w_img = frame.shape[:2]
                    # 左眼虹膜检测
                    left_iris_points = []
                    for idx in self.LEFT_IRIS_INDICES:
                        lm = face_landmarks.landmark[idx]
                        x_ = int(lm.x * w_img)
                        y_ = int(lm.y * h_img)
                        left_iris_points.append((x_, y_))
                    # 右眼虹膜检测
                    right_iris_points = []
                    for idx in self.RIGHT_IRIS_INDICES:
                        lm = face_landmarks.landmark[idx]
                        x_ = int(lm.x * w_img)
                        y_ = int(lm.y * h_img)
                        right_iris_points.append((x_, y_))
                    # 左眼轮廓检测
                    left_eye_points = []
                    for idx in self.LEFT_EYE_INDICES:
                        lm = face_landmarks.landmark[idx]
                        left_eye_points.append((int(lm.x * w_img), int(lm.y * h_img)))
                    # 右眼轮廓检测
                    right_eye_points = []
                    for idx in self.RIGHT_EYE_INDICES:
                        lm = face_landmarks.landmark[idx]
                        right_eye_points.append((int(lm.x * w_img), int(lm.y * h_img)))
                    # 绘制虹膜和眼睛轮廓
                    if len(left_iris_points) >= 5:
                        left_iris_ellipse = cv2.fitEllipse(np.array(left_iris_points))
                        cv2.ellipse(display_frame, left_iris_ellipse, (255, 255, 0), 1)
                    if len(right_iris_points) >= 5:
                        right_iris_ellipse = cv2.fitEllipse(np.array(right_iris_points))
                        cv2.ellipse(display_frame, right_iris_ellipse, (255, 255, 0), 1)
                    if len(left_eye_points) >= 5:
                        left_eye_ellipse = cv2.fitEllipse(np.array(left_eye_points))
                        cv2.ellipse(display_frame, left_eye_ellipse, (0, 255, 0), 1)
                    if len(right_eye_points) >= 5:
                        right_eye_ellipse = cv2.fitEllipse(np.array(right_eye_points))
                        cv2.ellipse(display_frame, right_eye_ellipse, (0, 255, 0), 1)
                    # 计算相对位置（加健壮性判断）
                    if left_iris_ellipse is not None and left_eye_ellipse is not None:
                        left_eye_center = left_eye_ellipse[0]
                        left_iris_center = left_iris_ellipse[0]
                        left_eye_major = left_eye_ellipse[1][0]
                        left_eye_minor = left_eye_ellipse[1][1]
                        if left_eye_major != 0 and left_eye_minor != 0:
                            left_relative = (
                                (left_iris_center[0] - left_eye_center[0]) / (left_eye_major / 2),
                                (left_iris_center[1] - left_eye_center[1]) / (left_eye_minor / 2)
                            )
                    if right_iris_ellipse is not None and right_eye_ellipse is not None:
                        right_eye_center = right_eye_ellipse[0]
                        right_iris_center = right_iris_ellipse[0]
                        right_eye_major = right_eye_ellipse[1][0]
                        right_eye_minor = right_eye_ellipse[1][1]
                        if right_eye_major != 0 and right_eye_minor != 0:
                            right_relative = (
                                (right_iris_center[0] - right_eye_center[0]) / (right_eye_major / 2),
                                (right_iris_center[1] - right_eye_center[1]) / (right_eye_minor / 2)
                            )
            # 标定阶段保存基线
            if not self.calibration_complete:
                if self.is_calibrating and left_relative and right_relative:
                    if not hasattr(self, '_eye_relative_samples'):
                        self._eye_relative_samples = []
                    self._eye_relative_samples.append({'left': left_relative, 'right': right_relative})
                if self.is_calibrating is False and hasattr(self, '_eye_relative_samples') and len(self._eye_relative_samples) > 0:
                    # 标定完成，取平均作为基线
                    left_x = np.mean([s['left'][0] for s in self._eye_relative_samples])
                    left_y = np.mean([s['left'][1] for s in self._eye_relative_samples])
                    right_x = np.mean([s['right'][0] for s in self._eye_relative_samples])
                    right_y = np.mean([s['right'][1] for s in self._eye_relative_samples])
                    self.eye_relative_baseline = {'left': (left_x, left_y), 'right': (right_x, right_y)}
                    del self._eye_relative_samples
            # 检测阶段分心判断
            distracted = False
            if self.calibration_complete and self.eye_relative_baseline and left_relative and right_relative:
                # 计算欧氏距离
                l_dist = np.sqrt((left_relative[0] - self.eye_relative_baseline['left'][0]) ** 2 + (left_relative[1] - self.eye_relative_baseline['left'][1]) ** 2)
                r_dist = np.sqrt((right_relative[0] - self.eye_relative_baseline['right'][0]) ** 2 + (right_relative[1] - self.eye_relative_baseline['right'][1]) ** 2)
                if l_dist > 0.25 or r_dist > 0.25:
                    # 偏移，开始计时
                    if self.distract_start_time is None:
                        self.distract_start_time = current_time
                    elif current_time - self.distract_start_time >= 3.0:
                        distracted = True
                else:
                    # 未偏移，重置计时
                    self.distract_start_time = None
            else:
                self.distract_start_time = None
            self.distracted = distracted
            if distracted:
                cv2.putText(display_frame, "Distracted!", (text_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            

            
            # 显示疲劳检测信息
            if fatigue_warning:
                cv2.putText(display_frame, "FATIGUE WARNING!", (text_x, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            elif eye_closed_frames > 0:
                cv2.putText(display_frame, f"Eyes closed: {eye_closed_frames} frames", (text_x, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif yawn_frames > 0:
                cv2.putText(display_frame, f"Yawn detected: {yawn_frames} frames", (text_x, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            # 显示视频画面
            if show_video:
                cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
                cv2.imshow('Emotion Detection', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    break

        # 释放资源
        if self.cap is not None:
            self.cap.release()
        if show_video:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    def print_result(result):
        """示例回调函数，打印检测结果"""
        print(f"\n检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"主要情感: {result['emotion']}")
        if result['micro_expression']:
            print(f"微表情: {result['micro_expression']}")
        print("各类别概率:")
        for emo, prob in result['all_probabilities'].items():
            print(f"  {emo}: {prob * 100:.2f}%")


    try:
        # 创建检测器实例
        detector = EmotionDetectorCamera(
            detection_interval=0.5,  # 每0.5秒检测一次
            callback=print_result,  # 设置回调函数
            use_chinese=False  # 使用英文显示
        )



        # 启动检测(显示视频窗口)
        if detector.start(show_video=True):
            print("按 'q' 键停止检测")
            while detector.is_running:
                time.sleep(0.1)
            detector.stop()
    except Exception as e:
        print(f"程序错误: {str(e)}")
