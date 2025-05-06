import threading
import time
import random

class PassiveEmotionMonitor:
    """监控摄像头情绪并在愤怒时自动讲笑话的被动监听器"""

    def __init__(self, chatbot, interval: float = 2.0):
        """
        Args:
            chatbot: 已初始化的 EmotionAwareStreamingChatbot 实例
            interval: 每隔多久检测一次情绪（秒）
        """
        self.chatbot = chatbot
        self.interval = interval
        self.running = False
        self.thread = None
        self.jokes = [
            "我刚刚试图给空气打电话，但它没信号。",
            "你知道程序员最怕什么吗？Bug吓！",
            "我昨天吃了一个回旋镖，结果它又回来了！",
            "为什么猫不玩电脑？因为它怕鼠标。",
            "我跟鱼聊天，它竟然对我说“水你在说什么？”"
        ]

    def start(self):
        """启动情绪检测线程"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop)
            self.thread.daemon = True
            self.thread.start()
            print("PassiveEmotionMonitor started.")

    def stop(self):
        """停止情绪检测"""
        self.running = False
        if self.thread:
            self.thread.join()
            print("PassiveEmotionMonitor stopped.")

    def _monitor_loop(self):
        """情绪检测主循环"""
        while self.running:
            try:
                emotion = self.chatbot.get_current_emotion()
                if emotion.lower() == "愤怒" or emotion.lower() == "angry":
                    print("检测到愤怒情绪，尝试讲个笑话缓解气氛...")
                    joke = random.choice(self.jokes)
                    self.chatbot.speak(joke)
                    time.sleep(10)  # 避免连续触发
                else:
                    time.sleep(self.interval)
            except Exception as e:
                print(f"Passive monitor error: {e}")
                time.sleep(self.interval)
