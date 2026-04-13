"""
Stability Tracking Module
Tracks face position stability for reliable recognition
"""

import numpy as np
from typing import List, Tuple


class StabilityTracker:
    """Tracks face stability for reliable recognition"""
    
    def __init__(self, config):
        self.config = config
        self.face_history: List[Tuple[int, int]] = []
        self.stable_count = 0
    
    def update(self, face: Tuple[int, int, int, int]) -> bool:
        """Update with new face position and check stability"""
        x, y, w, h = face
        center = (x + w // 2, y + h // 2)
        
        self.face_history.append(center)
        if len(self.face_history) > self.config.face_history_size:
            self.face_history.pop(0)
        
        if len(self.face_history) < 3:
            self.stable_count = 0
            return False
        
        if self._is_stable():
            self.stable_count += 1
        else:
            self.stable_count = 0
        
        return self.stable_count >= self.config.stability_frames
    
    def _is_stable(self) -> bool:
        """Check if recent positions indicate stability"""
        recent = self.face_history[-3:]
        max_distance = 0
        
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                dist = np.sqrt(
                    (recent[i][0] - recent[j][0]) ** 2 + 
                    (recent[i][1] - recent[j][1]) ** 2
                )
                max_distance = max(max_distance, dist)
        
        return max_distance < self.config.motion_threshold
    
    def reset(self) -> None:
        """Reset tracking state"""
        self.face_history.clear()
        self.stable_count = 0
    
    @property
    def is_stable(self) -> bool:
        """Check if currently stable"""
        return self.stable_count >= self.config.stability_frames