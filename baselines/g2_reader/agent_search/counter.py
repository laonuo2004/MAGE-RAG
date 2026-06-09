import os
from pathlib import Path

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

class ProcessSafeCounter:
    """Process-safe counter for tracking progress"""
    def __init__(self, save_dir: str):
        self.counter_file = Path(save_dir) / "progress_counter.txt"
        self.lock_file = Path(save_dir) / "progress_counter.lock"
    
    def increment(self, amount: int = 1):
        try:
            with open(self.lock_file, 'w') as lock:
                if HAS_FCNTL:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                
                current = 0
                if self.counter_file.exists():
                    try:
                        current = int(self.counter_file.read_text().strip())
                    except:
                        current = 0
                
                new_value = current + amount
                self.counter_file.write_text(str(new_value))
                
                if HAS_FCNTL:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                
                return new_value
        except Exception as e:
            print(f"Counter update failed: {e}")
            return -1
    
    def get_count(self):
        try:
            if self.counter_file.exists():
                return int(self.counter_file.read_text().strip())
            return 0
        except:
            return 0
