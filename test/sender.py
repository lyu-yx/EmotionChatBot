# sender.py
import time
import sys
import os

import SharedQueue

def sender():
    q = SharedQueue.SharedQueue()
    for i in range(5):
        msg = f"消息 {i}"
        print(f"[Sender] 放入: {msg}")
        q.put(msg)
        time.sleep(1)
