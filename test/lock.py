import threading

lock = threading.Lock()

def critical_section():
    print("Trying to acquire lock...")
    lock.acquire()
    print("Lock acquired!")
    # 模拟耗时操作
    import time; time.sleep(3)
    print("Releasing lock...")
    lock.release()

# 多线程测试
t1 = threading.Thread(target=critical_section)
t2 = threading.Thread(target=critical_section)

t1.start()
t2.start()
