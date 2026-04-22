"""
Configuration Module for Buddy System
Handles all configuration settings and environment variables
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional
import cv2


@dataclass
class Config:
    """Configuration management for the Buddy system"""
    
    # Model paths
    model_path: str = "vision/models/MobileFaceNet.onnx"
    cascade_path: str = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    # Camera settings (Pi optimized)
    camera_index: int = 0
    camera_width: int = 320
    camera_height: int = 240
    camera_fps: int = 15
    camera_buffer_size: int = 1
    
    # Recognition thresholds
    recognition_threshold: float = 0.5   # Back to original threshold
    confidence_threshold: float = 0.1   # Very low confidence requirement
    min_face_size: Tuple[int, int] = (40, 40)  # Smaller minimum size
    max_face_size: Tuple[int, int] = (600, 600)
    
    # Timing settings (Pi optimized)
    recognition_interval: float = 2.0
    sleep_threshold: int = 30
    wake_delay: float = 0.5
    frame_process_interval: float = 0.1
    voice_sleep_timeout: float = 3.5
    
    # Motion detection
    motion_threshold: int = 80  # Much more lenient movement tolerance
    stability_frames: int = 1   # Only 1 frame needed for stability
    face_history_size: int = 3  # Smaller history for faster response
    
    # Sleep/Wake settings
    enable_sleep_mode: bool = True
    sleep_message: str = "💤 Going to sleep... No one around."
    wake_message: str = " Someone's here!"
    
    # Voice-based sleep/wake settings
    max_listening_attempts: int = 4  # Increased from 3 to 4
    wake_word: str = "hey0-buddy"
    voice_sleep_message: str = "💤 Going to sleep... Say 'Hey Buddy' to wake me up."
    
    # Logging
    log_level: str = "WARNING"  # Changed from INFO to WARNING
    log_file: Optional[str] = "buddy.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # LLM Service Configuration
    llm_service_url: str = os.getenv("LLM_SERVICE_URL", "http://buddypc.local:8000")

    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        config = cls()
        config.model_path = os.getenv('BUDDY_MODEL_PATH', config.model_path)
        config.camera_index = int(os.getenv('BUDDY_CAMERA_INDEX', str(config.camera_index)))
        config.log_level = os.getenv('BUDDY_LOG_LEVEL', config.log_level)
        config.llm_service_url = os.getenv('LLM_SERVICE_URL', config.llm_service_url)
        return config
