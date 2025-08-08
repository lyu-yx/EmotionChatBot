# shared_queue.py
import queue

class SharedQueue:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.queue = queue.Queue()
        return cls._instance

    def put(self, item):
        self.queue.put(item)

    def get(self, block=True, timeout=None):
        return self.queue.get(block=block, timeout=timeout)

    def empty(self):
        return self.queue.empty()
