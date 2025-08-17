import snowboydecoder
import sys
import signal
import os

def detected_callback():
    print("Wake word detected! Exiting...")
    snowboydecoder.play_audio_file()  # 播放提示音（可选）
    sys.exit(0)  # 检测到关键词后退出

if __name__ == "__main__":
    model = "/home/adminpc/桌面/123/hotword.pmdl"
    detector = snowboydecoder.HotwordDetector(model, sensitivity=0.5)
    print('Listening... Press Ctrl+C to exit')
    detector.start(detected_callback=detected_callback)  # 修改回调函数
    detector.terminate()
