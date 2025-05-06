# shared_queue.py
import queue
size = 20
class SharedQueue:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.queue = queue.Queue(size)
        return cls._instance

    def put(self, item):
        self.queue.put(item)

    def get(self, block=True, timeout=None):
        return self.queue.get(block=block, timeout=timeout)

    def empty(self):
        return self.queue.empty()
    def peek(self):
        if self.queue.queue:
            return self.queue.queue[-1]#后续代码中请注意：不加锁可能有线程安全问题
    def clear(queue):
        while not queue.empty():
            queue.get()
