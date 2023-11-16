import threading
import collections


# reference: https://gist.github.com/vitaliyp/6d54dd76ca2c3cdfc1149d33007dc34a
class FIFOLock(object):
    """
    先进先出（FIFO）锁
    """
    def __init__(self):
        # 用于实际控制对共享资源的访问的锁
        self._lock = threading.Lock()
        # _inner_lock 的主要作用是保护对内部数据结构（例如 self._pending_threads）的并发访问。
        self._inner_lock = threading.Lock()
        # 用于存储等待获取锁的线程
        self._pending_threads = collections.deque()

    def acquire(self, blocking=True):
        """
        尝试获取锁。如果锁已经被获取，则返回 False。
        如果锁未被获取，并且设置为非阻塞模式（blocking=False），则返回 False。
        否则，将创建一个事件，并将其添加到 self._pending_threads 中，然后等待该事件被触发（即等待锁被释放）。
        一旦事件被触发，将再次尝试获取锁，并返回结果。
        """
        with self._inner_lock:
            lock_acquired = self._lock.acquire(False)
            if lock_acquired:
                return True
            elif not blocking:
                return False

            release_event = threading.Event()
            self._pending_threads.append(release_event)

        release_event.wait()
        return self._lock.acquire()

    def release(self):
        """ 释放锁 """
        with self._inner_lock:
            if self._pending_threads:
                # 存在等待的线程，那么将创建一个新的事件，并将其添加到队列的前面，然后触发这个事件。这将使得最早等待的线程能够获取到锁。
                release_event = self._pending_threads.popleft()
                release_event.set()

            self._lock.release()

    # 当使用 with 语句使用 FIFOLock 时，acquire 方法会被调用以尝试获取
    __enter__ = acquire

    # release 方法则会在 with 语句结束时被调用以释放锁。
    def __exit__(self, t, v, tb):
        self.release()
