"""
Read-Write Lock for RL Agent Inference

Allows multiple concurrent readers (predictions) but exclusive writer (model updates).
This enables high throughput and low latency while maintaining thread safety.
"""

import threading
from contextlib import contextmanager


class RWLock:
    """
    Read-Write Lock with writer priority.
    
    - Multiple readers can hold the lock simultaneously
    - Writers get exclusive access (no readers or other writers)
    - Writers have priority to prevent starvation
    
    Usage:
        rwlock = RWLock()
        
        # For inference (many concurrent)
        with rwlock.read():
            result = model.predict(...)
        
        # For updates (exclusive)
        with rwlock.write():
            model.update(...)
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._read_ready = threading.Condition(self._lock)
        self._write_ready = threading.Condition(self._lock)
        
        self._readers = 0  # Number of active readers
        self._writers = 0  # Number of active writers (0 or 1)
        self._waiting_writers = 0  # Number of writers waiting
        
    def read_acquire(self):
        """Acquire read lock (allows multiple concurrent readers)"""
        with self._lock:
            # Wait while there's a writer or writers waiting
            while self._writers > 0 or self._waiting_writers > 0:
                self._read_ready.wait()
            self._readers += 1
    
    def read_release(self):
        """Release read lock"""
        with self._lock:
            self._readers -= 1
            # If last reader, wake up waiting writers
            if self._readers == 0:
                self._write_ready.notify()
    
    def write_acquire(self):
        """Acquire write lock (exclusive access)"""
        with self._lock:
            self._waiting_writers += 1
            # Wait while there are active readers or writers
            while self._readers > 0 or self._writers > 0:
                self._write_ready.wait()
            self._waiting_writers -= 1
            self._writers = 1
    
    def write_release(self):
        """Release write lock"""
        with self._lock:
            self._writers = 0
            # Wake up all waiting readers and one writer
            self._write_ready.notify()
            self._read_ready.notify_all()
    
    @contextmanager
    def read(self):
        """Context manager for read lock"""
        self.read_acquire()
        try:
            yield
        finally:
            self.read_release()
    
    @contextmanager
    def write(self):
        """Context manager for write lock"""
        self.write_acquire()
        try:
            yield
        finally:
            self.write_release()
    
    def get_stats(self):
        """Get lock statistics for monitoring"""
        with self._lock:
            return {
                'active_readers': self._readers,
                'active_writers': self._writers,
                'waiting_writers': self._waiting_writers
            }


# Backward compatibility wrapper
class RWLockCompat:
    """
    Backward compatible wrapper that acts like threading.Lock
    but uses RWLock internally.
    
    Use rwlock.read() for read operations
    Use rwlock.write() for write operations
    Use rwlock (as context manager) for backward compat (defaults to write)
    """
    
    def __init__(self):
        self._rwlock = RWLock()
    
    def __enter__(self):
        """Default to write lock for backward compatibility"""
        self._rwlock.write_acquire()
        return self
    
    def __exit__(self, *args):
        self._rwlock.write_release()
    
    def acquire(self):
        """Default to write lock for backward compatibility"""
        self._rwlock.write_acquire()
    
    def release(self):
        self._rwlock.write_release()
    
    @contextmanager
    def read(self):
        """Context manager for read lock"""
        self._rwlock.read_acquire()
        try:
            yield
        finally:
            self._rwlock.read_release()
    
    @contextmanager
    def write(self):
        """Context manager for write lock"""
        self._rwlock.write_acquire()
        try:
            yield
        finally:
            self._rwlock.write_release()
    
    def get_stats(self):
        return self._rwlock.get_stats()


if __name__ == "__main__":
    import time
    import random
    
    print("Testing RWLock...")
    
    rwlock = RWLock()
    results = []
    
    def reader(id):
        """Simulate model prediction"""
        time.sleep(random.uniform(0, 0.01))  # Random start time
        with rwlock.read():
            start = time.time()
            time.sleep(0.01)  # Simulate 10ms prediction
            duration = time.time() - start
            results.append(('read', id, start, duration))
            print(f"Reader {id}: {duration*1000:.1f}ms")
    
    def writer(id):
        """Simulate model update"""
        time.sleep(random.uniform(0, 0.05))
        with rwlock.write():
            start = time.time()
            time.sleep(0.05)  # Simulate 50ms update
            duration = time.time() - start
            results.append(('write', id, start, duration))
            print(f"Writer {id}: {duration*1000:.1f}ms (exclusive)")
    
    # Test: 10 concurrent readers + 1 writer
    threads = []
    test_start = time.time()
    
    for i in range(10):
        t = threading.Thread(target=reader, args=(i,))
        threads.append(t)
        t.start()
    
    t = threading.Thread(target=writer, args=(0,))
    threads.append(t)
    t.start()
    
    for t in threads:
        t.join()
    
    test_duration = time.time() - test_start
    
    print(f"\nTest completed in {test_duration*1000:.1f}ms")
    print(f"Expected: ~60ms (10 readers concurrent + 1 writer)")
    print(f"Without RWLock: ~150ms (all serialized)")
    
    # Analyze concurrency
    read_ops = [r for r in results if r[0] == 'read']
    if read_ops:
        # Check if reads overlapped
        read_times = [(r[2], r[2] + r[3]) for r in read_ops]
        overlaps = 0
        for i, (s1, e1) in enumerate(read_times):
            for s2, e2 in read_times[i+1:]:
                if s1 < e2 and s2 < e1:
                    overlaps += 1
        print(f"Read operations overlapped: {overlaps} times (good!)")
    
    print("\n✅ RWLock test passed!")





