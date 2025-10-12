import time
from logger import logger

# ============================================================================
# Episode Tracker
# ============================================================================

class EpisodeTracker:
    """
    Tracks episode boundaries for proper credit assignment.
    
    Episodes define the scope of multi-step returns:
    - Time-based: All requests in 1-second window share credit
    - Provides done flags for TD learning
    """
    def __init__(self, episode_duration=1.0):
        """
        Args:
            episode_duration: Episode length in seconds
        """
        self.episode_duration = episode_duration
        self.episode_start_time = time.time()
        self.episode_id = 0
        self.episode_request_count = 0
        
    def check_episode_end(self):
        """Returns True if current episode should end"""
        elapsed = time.time() - self.episode_start_time
        return elapsed >= self.episode_duration
    
    def reset_episode(self):
        """Start new episode"""
        self.episode_start_time = time.time()
        self.episode_id += 1
        self.episode_request_count = 0
        logger.info(f"📍 Episode {self.episode_id} started")
    
    def increment_request(self):
        """Track request count in episode"""
        self.episode_request_count += 1