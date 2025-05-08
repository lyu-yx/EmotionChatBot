import time
import sys
import os

import SharedQueue


def receiver():
    q = SharedQueue.SharedQueue()
    while True:
        if not q.empty():
            msg = q.get()
            print(f"[Receiver] 取出: {msg}")
        time.sleep(0.5)
