import cv2
import numpy as np
import time

class TestVisualization:
    def __init__(self):
        self.emotion_score = 0
        
    def _draw_gauge(self, frame, score, x, y, radius=60):
        """绘制油表盘样式的评分表"""
        # 绘制外圆
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)
        
        # 绘制彩色背景（渐变效果）
        for i in range(-10, 11):
            angle = np.pi * (i + 10) / 20  # 将-10到10映射到0到π
            start_x = x + int((radius - 15) * np.cos(angle))
            start_y = y - int((radius - 15) * np.sin(angle))
            end_x = x + int(radius * np.cos(angle))
            end_y = y - int(radius * np.sin(angle))
            
            # 计算颜色渐变（-10红色，0黄色，10绿色）
            if i <= 0:
                # 从红色到黄色
                ratio = (i + 10) / 10
                color = (
                    int(255 * (1 - ratio)),  # B
                    int(255 * ratio),        # G
                    255                      # R
                )
            else:
                # 从黄色到绿色
                ratio = i / 10
                color = (
                    0,                       # B
                    255,                     # G
                    int(255 * (1 - ratio))   # R
                )
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 3)
        
        # 绘制刻度
        for i in range(-10, 11, 2):
            angle = np.pi * (i + 10) / 20  # 将-10到10映射到0到π
            start_x = x + int((radius - 10) * np.cos(angle))
            start_y = y - int((radius - 10) * np.sin(angle))
            end_x = x + int(radius * np.cos(angle))
            end_y = y - int(radius * np.sin(angle))
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (100, 100, 100), 2)
        
        # 绘制指针
        pointer_angle = np.pi * (score + 10) / 20
        pointer_x = x + int((radius - 20) * np.cos(pointer_angle))
        pointer_y = y - int((radius - 20) * np.sin(pointer_angle))
        
        # 根据评分选择指针颜色
        if score <= -5:
            pointer_color = (0, 0, 255)  # 红色
        elif score <= 0:
            pointer_color = (0, 255, 255)  # 黄色
        elif score <= 5:
            pointer_color = (0, 255, 128)  # 浅绿色
        else:
            pointer_color = (0, 255, 0)  # 绿色
        
        cv2.line(frame, (x, y), (pointer_x, pointer_y), pointer_color, 4)
        
        # 绘制中心点
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        
        # 绘制评分文字
        score_text = f"{score:+d}"
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
        cv2.putText(frame, score_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # 绘制标题
        title = "Emotion Score"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        title_x = x - title_size[0] // 2
        title_y = y + radius + 20
        cv2.putText(frame, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_hexagon_chart(self, frame, model_scores, x, y, size=80):
        """绘制六芒星图显示所有情感的加权平均置信度"""
        # 绘制外圆
        cv2.circle(frame, (x, y), size, (100, 100, 100), 2)
        
        # 绘制同心圆（表示不同的置信度级别）
        for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
            radius = int(size * level)
            cv2.circle(frame, (x, y), radius, (50, 50, 50), 1)
        
        # 计算所有情感的加权平均置信度
        emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        weights = [0.4, 0.3, 0.3]  # 对应MODEL_WEIGHTS中的权重
        models = ['deepface', 'fer', 'vgg']
        
        # 计算每个情感的加权平均置信度
        emotion_confidences = {}
        for emotion in emotion_classes:
            total_confidence = 0
            valid_models = 0
            for model, weight in zip(models, weights):
                if model in model_scores:
                    confidence = model_scores[model]['scores'].get(emotion, 0)
                    total_confidence += confidence * weight
                    valid_models += 1
            
            if valid_models > 0:
                avg_confidence = total_confidence / sum(weights[:valid_models])
                emotion_confidences[emotion] = avg_confidence
        
        # 定义固定的情感位置（6个情感，60度间隔）
        emotion_positions = {
            'happy': 0,      # 12点钟方向
            'surprise': 60,  # 2点钟方向
            'fear': 120,     # 4点钟方向
            'sad': 180,      # 6点钟方向
            'angry': 240,    # 8点钟方向
            'disgust': 300   # 10点钟方向
        }
        emotion_colors = {
            'happy': (0, 255, 0),      # 绿色
            'surprise': (255, 255, 0), # 黄色
            'fear': (128, 0, 128),     # 紫色
            'sad': (0, 0, 255),        # 蓝色
            'angry': (0, 0, 0),        # 黑色
            'disgust': (128, 128, 128) # 灰色
        }
        
        # 存储所有点用于绘制覆盖面积
        all_points = []
        
        # 绘制所有情感的置信度（固定位置）
        for emotion, angle in emotion_positions.items():
            if emotion in emotion_confidences:
                confidence = emotion_confidences[emotion]
                color = emotion_colors[emotion]
                rad = np.radians(angle)
                radius = int(size * confidence)
                point_x = x + int(radius * np.cos(rad))
                point_y = y - int(radius * np.sin(rad))
            
            # 存储点
            all_points.append((point_x, point_y))
            
            # 绘制点
            cv2.circle(frame, (point_x, point_y), 5, color, -1)
            
            # 绘制连接线
            cv2.line(frame, (x, y), (point_x, point_y), color, 3)
            
            # 绘制情感名称（更明显）
            emotion_text = emotion.upper()
            text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x + int((size + 20) * np.cos(rad)) - text_size[0] // 2
            text_y = y - int((size + 20) * np.sin(rad)) + text_size[1] // 2
            cv2.putText(frame, emotion_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 绘制置信度数值
            conf_text = f"{confidence:.2f}"
            conf_text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            conf_text_x = point_x - conf_text_size[0] // 2
            conf_text_y = point_y - 10
            cv2.putText(frame, conf_text, (conf_text_x, conf_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 如果有足够的点，绘制红色填充区域
        if len(all_points) >= 3:
            # 添加中心点
            all_points.append((x, y))
            
            # 转换为numpy数组
            points = np.array(all_points, dtype=np.int32)
            
            # 绘制红色填充区域
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], (0, 0, 255))  # 红色填充
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # 绘制边界线
            cv2.polylines(frame, [points], True, (255, 255, 255), 2)
        
        # 计算并显示总体加权平均置信度
        total_avg_confidence = sum(emotion_confidences.values()) / len(emotion_confidences) if emotion_confidences else 0
        
        # 在中心显示总体平均置信度
        avg_text = f"Avg: {total_avg_confidence:.2f}"
        avg_text_size = cv2.getTextSize(avg_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        avg_text_x = x - avg_text_size[0] // 2
        avg_text_y = y + avg_text_size[1] // 2
        cv2.putText(frame, avg_text, (avg_text_x, avg_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 绘制标题
        title = "Emotion Confidence"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        title_x = x - title_size[0] // 2
        title_y = y - size - 25
        cv2.putText(frame, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def test_visualization(self):
        """测试可视化功能"""
        # 创建测试画面
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 模拟模型分数
        model_scores = {
            'deepface': {
                'emotion': 'happy',
                'scores': {'happy': 0.8, 'sad': 0.1, 'angry': 0.05, 'neutral': 0.05}
            },
            'fer': {
                'emotion': 'happy',
                'scores': {'happy': 0.7, 'sad': 0.15, 'angry': 0.1, 'neutral': 0.05}
            },
            'vgg': {
                'emotion': 'happy',
                'scores': {'happy': 0.9, 'sad': 0.05, 'angry': 0.03, 'neutral': 0.02}
            }
        }
        
        # 测试不同评分
        scores_to_test = [-10, -5, 0, 5, 10]
        
        for score in scores_to_test:
            # 清空画面
            frame.fill(0)
            
            # 绘制仪表盘（左下角）
            self._draw_gauge(frame, score, 100, 400, 50)
            
            # 绘制六芒星图（右上角）
            self._draw_hexagon_chart(frame, model_scores, 540, 100, 60)
            
            # 显示当前测试的评分
            cv2.putText(frame, f"Testing Score: {score}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示画面
            cv2.imshow('Visualization Test', frame)
            
            # 等待2秒
            if cv2.waitKey(2000) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test = TestVisualization()
    test.test_visualization() 