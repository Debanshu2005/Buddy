"""
Raspberry Pi-safe face detector based on OpenCV Haar cascades.
"""

import cv2
import logging
from typing import List, Optional, Tuple

import numpy as np


class FaceDetector:
    """Detect faces using a lightweight OpenCV cascade."""

    def __init__(self, cascade_path: str, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cascade_path = cascade_path or (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._cascade = None
        self._init_detector()

    def _init_detector(self):
        try:
            self._cascade = cv2.CascadeClassifier(self.cascade_path)
            if self._cascade.empty():
                raise RuntimeError(f"Failed to load cascade at {self.cascade_path}")
            print("✅ OpenCV face detector ready")
        except Exception as e:
            self.logger.error(f"Face detector init failed: {e}")
            self._cascade = None
            print(f"⚠️ Face detector unavailable: {e}")

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self._cascade is None:
            return []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=self.config.min_face_size,
                maxSize=self.config.max_face_size,
            )
            return [
                (int(x), int(y), int(w), int(h))
                for (x, y, w, h) in faces
                if w >= self.config.min_face_size[0]
                and h >= self.config.min_face_size[1]
            ]
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []

    def get_largest_face(
        self, faces: List[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        if not faces:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        return image
