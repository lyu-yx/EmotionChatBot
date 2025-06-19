# This class was originally written to address potential PyAudio conflicts, but it may no longer be necessary due to the AudioManager. It can be optimized in the future.
import threading
class SharedLock:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._lock = threading.Lock()
        return cls._instance

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
