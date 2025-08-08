# main.py
import threading
from sender import sender
from receiver import receiver

if __name__ == "__main__":
    # 创建并启动发送线程
    sender_thread = threading.Thread(target=sender)
    sender_thread.daemon = True  # 设置为守护线程，主线程结束时自动退出
    sender_thread.start()

    # 创建并启动接收线程
    receiver_thread = threading.Thread(target=receiver)
    receiver_thread.daemon = True  # 设置为守护线程，主线程结束时自动退出
    receiver_thread.start()

    # 等待发送线程完成（接收线程会一直运行）
    sender_thread.join()
