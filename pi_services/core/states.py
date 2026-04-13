"""
State Management Module for Buddy System
Defines system states and transitions
"""

from enum import Enum
from typing import Optional
import logging


class BuddyState(Enum):
    """System states for the Buddy application"""
    INITIALIZING = "initializing"
    SLEEPING = "sleeping"
    VOICE_SLEEPING = "voice_sleeping"  # New state for voice-based sleep
    WAKING = "waking"
    ACTIVE = "active"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class StateManager:
    """Manages state transitions with logging"""
    
    def __init__(self):
        self._state = BuddyState.INITIALIZING
        self.logger = logging.getLogger(__name__)
        self.previous_state: Optional[BuddyState] = None
    
    @property
    def state(self) -> BuddyState:
        return self._state
    
    @state.setter
    def state(self, new_state: BuddyState):
        if self._state != new_state:
            self.previous_state = self._state
            self._state = new_state
            self.logger.info(f"State transition: {self.previous_state.value} -> {new_state.value}")
    
    def is_sleeping(self) -> bool:
        return self._state in [BuddyState.SLEEPING, BuddyState.VOICE_SLEEPING]
    
    def is_voice_sleeping(self) -> bool:
        return self._state == BuddyState.VOICE_SLEEPING
    
    def is_active(self) -> bool:
        return self._state == BuddyState.ACTIVE
    
    def is_waking(self) -> bool:
        return self._state == BuddyState.WAKING
    
    def can_process_faces(self) -> bool:
        return self._state in [BuddyState.ACTIVE, BuddyState.WAKING, BuddyState.PROCESSING]