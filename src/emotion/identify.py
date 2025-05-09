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
from PIL import ImageFont, ImageDraw, Image
import dlib

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cv2_putText_cn(img, text, position, font_path="simhei.ttf", font_size=32, color=(0, 255, 0)):
    """
    在 OpenCV 图像上绘制支持中文的文字（使用Pillow）。

    参数：
        img         - OpenCV图像（numpy数组）
        text        - 要绘制的文本（可以是中文/emoji）
        position    - 文本起始位置 (x, y)
        font_path   - 字体文件路径，默认使用黑体（simhei.ttf）
        font_size   - 字体大小
        color       - 文本颜色，(B, G, R)

    返回：
        带文字的新图像（OpenCV格式）
    """
    # 转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 加载字体（需确保字体文件存在）
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        raise RuntimeError(f"字体加载失败，请确保字体文件路径正确：{e}")

    # 绘制文字
    draw.text(position, text, font=font, fill=color[::-1])  # RGB转BGR

    # 转换回OpenCV图像
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


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
        self.detector = dlib.get_frontal_face_detector()
        # dlib的68点模型
        self.predictor = dlib.shape_predictor("./src/emotion/shape_predictor_68_face_landmarks.dat")

        # 创建摄像头对象
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 480)
        self.cnt = 0

        # 表情阈值参数
        self.thresholds = {
            'mouth_higth_happy': 0.028,
            'mouth_higth_amazing': 0.025,
            'eye_hight_amazing': 0.053,
            'brow_k_angry': -0.1,
            'mouth_higth_sad': -0.05,
            'brow_width_disgust': 0.5,
            'eye_hight_fear': 0.05,
            'mouth_core_width': 0.06,
            'mouth_core_hight': 0.005
        }

    def get_landmark_features(self, shape, face_rect):
        """提取面部特征点特征"""
        face_width = face_rect.right() - face_rect.left()
        face_height = face_rect.bottom() - face_rect.top()

        # 嘴巴特征
        mouth_width = (shape.part(54).x - shape.part(48).x) / face_width
        mouth_higth = (shape.part(51).y - shape.part(57).y) / face_width
        mouth_core_width = (shape.part(59).x - shape.part(48).x) / face_width
        mouth_core_hight = (shape.part(48).y - shape.part(60).y) / face_width
        # 眉毛特征
        brow_sum = 0  # 高度之和
        frown_sum = 0  # 两边眉毛距离之和
        line_brow_x = []
        line_brow_y = []

        for j in range(17, 21):
            brow_sum += (shape.part(j).y - face_rect.top()) + (shape.part(j + 5).y - face_rect.top())
            frown_sum += shape.part(j + 5).x - shape.part(j).x
            line_brow_x.append(shape.part(j).x)
            line_brow_y.append(shape.part(j).y)

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
        eye_sum = (shape.part(41).y - shape.part(37).y + shape.part(40).y - shape.part(38).y +
                   shape.part(47).y - shape.part(43).y + shape.part(46).y - shape.part(44).y)
        eye_hight = (eye_sum / 4) / face_width

        # 鼻子皱起程度 (厌恶表情)
        nose_wrinkling = (shape.part(31).y - shape.part(27).y) / face_height

        return {
            'mouth_width': mouth_width,
            'mouth_higth': mouth_higth,
            'brow_k': brow_k,
            'brow_hight': brow_hight,
            'brow_width': brow_width,
            'eye_hight': eye_hight,
            'nose_wrinkling': nose_wrinkling,
            'mouth_core_width': mouth_core_width,
            'mouth_core_hight': mouth_core_hight
        }

    def detect_emotion(self, features):
        """根据特征判断表情"""
        thresholds = self.thresholds

        # 惊讶 (眼睛睁大+嘴巴张大)
        if (features['eye_hight'] >= thresholds['eye_hight_amazing']):
            return "amazing"
        elif (features['mouth_higth'] >= thresholds['mouth_higth_amazing'] and
              features['eye_hight'] >= thresholds['eye_hight_amazing']):
            return "amazing"

        # 开心 (嘴巴张大但眼睛不一定)
        elif features['mouth_core_width'] >= thresholds['mouth_core_width']:
            return "happy"

        # 生气 (眉毛内聚且下压)
        elif features['brow_k'] <= thresholds['brow_k_angry']:
            return "angry"

        # 悲伤 (眉毛外角上扬)
        elif features['mouth_core_hight'] >= thresholds['mouth_core_hight']:
            return "sad"

        # 厌恶 (鼻子皱起+眉毛压低)
        elif (features['nose_wrinkling'] < 0.25) :
            return "disgust"

        # 恐惧 (眼睛睁大+眉毛上扬)
        elif (features['eye_hight'] >= thresholds['eye_hight_fear'] and
              features['brow_hight'] > 0.25):
            return "fear"

        # 默认自然表情
        else:
            return "nature"

    def learning_face(self):
        while self.cap.isOpened():
            flag, im_rd = self.cap.read()
            k = cv2.waitKey(1)

            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
            faces = self.detector(img_gray, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX

            if len(faces) != 0:
                for k, d in enumerate(faces):
                    # 用红色矩形框出人脸
                    cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))

                    # 使用预测器得到68点数据的坐标
                    shape = self.predictor(im_rd, d)

                    # 圆圈显示每个特征点
                    for i in range(68):
                        cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)

                    # 提取特征
                    features = self.get_landmark_features(shape, d)

                    # 检测表情
                    emotion = self.detect_emotion(features)

                    # 显示表情结果
                    cv2.putText(im_rd, emotion, (d.left(), d.bottom() + 20),
                                font, 0.8, (0, 0, 255), 2, 4)

                # 标出人脸数
                cv2.putText(im_rd, "Faces: " + str(len(faces)), (20, 50),
                            font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(im_rd, "No Face", (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # 添加说明
            im_rd = cv2.putText(im_rd, "S: screenshot", (20, 400),
                                font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            im_rd = cv2.putText(im_rd, "Q: quit", (20, 450),
                                font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            # 按下s键截图保存
            if k == ord('s'):
                self.cnt += 1
                cv2.imwrite("screenshoot" + str(self.cnt) + ".jpg", im_rd)

            # 按下q键退出
            if k == ord('q'):
                break

            # 窗口显示
            cv2.imshow("Facial Expression Recognition", im_rd)

        # 释放资源q
        self.cap.release()
        cv2.destroyAllWindows()

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
        last_detection_time = 0

        while self.is_running and self.cap is not None:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Unable to get video frame")
                break

            current_time = time.time()

            # 在每一帧都绘制检测框和特征点
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(img_gray, 0)

            if len(faces) != 0:
                for k, d in enumerate(faces):
                    # 用红色矩形框出人脸
                    cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)

                    # 使用预测器得到68点数据的坐标
                    shape = self.predictor(frame, d)

                    # 绘制68个特征点（绿色小圆点）
                    for i in range(68):
                        cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1)

            # 只在检测间隔时间进行表情检测
            if current_time - last_detection_time >= self.detection_interval:
                try:
                    # 检测表情
                    if len(faces) > 0:
                        for k, d in enumerate(faces):
                            shape = self.predictor(frame, d)
                            features = self.get_landmark_features(shape, d)
                            emotion = self.detect_emotion(features)

                            # 在脸上方显示检测到的表情
                            cv2.putText(frame, emotion, (d.left(), d.top() - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                            # Update latest result
                            with self.lock:
                                self.latest_result = {
                                    "emotion": emotion,
                                    "emotion_index": self.EMOTION_CLASSES.index(emotion) if emotion in self.EMOTION_CLASSES else 6,
                                    "probability": 1.0,  # 特征点方法没有概率，设为1.0
                                    "all_probabilities": {emo: 1.0 if emo == emotion else 0.0 for emo in self.emotion_classes},
                                    "timestamp": time.time()
                                }

                                # Call callback function (if any)
                                if self.callback is not None:
                                    self.callback(self.latest_result)

                    last_detection_time = current_time
                except Exception as e:
                    print(f"Emotion detection error: {str(e)}")

            # 如果需要显示视频
            if show_video:
                cv2.imshow('Emotion Detection', frame)

                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    break

        # 释放资源
        if self.cap is not None:
            self.cap.release()
        if show_video:
            cv2.destroyAllWindows()


# Example usage when running this script directly
if __name__ == "__main__":
    def print_emotion(result):
        print(f"\nDetection time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Emotion: {result['emotion']} (Confidence: {result['probability'] * 100:.2f}%)")
        print("Probabilities for all categories:")
        for emo, prob in result['all_probabilities'].items():
            print(f"  {emo}: {prob * 100:.2f}%")


    try:
        detector = EmotionDetectorCamera(
            detection_interval=0.5,  # 每0.5秒检测一次
            callback=print_emotion,
            use_chinese=False
        )

        if detector.start(show_video=True):
            print("Press 'q' to stop detection")

            try:
                while detector.is_running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Detection interrupted by user")
            finally:
                detector.stop()

    except Exception as e:
        print(f"Program error: {str(e)}")