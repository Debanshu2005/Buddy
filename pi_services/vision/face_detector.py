"""
Face Detection Module
Uses InsightFace RetinaFace detector
"""

import numpy as np
import logging
from typing import List, Tuple, Optional


class FaceDetector:
    """Handles face detection using InsightFace RetinaFace"""

    def __init__(self, cascade_path: str, config):
        # cascade_path kept for API compatibility but not used
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._app = None
        self._init_detector()

    def _init_detector(self):
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(
                name="buffalo_sc",
                allowed_modules=["detection"],
                providers=["CPUExecutionProvider"]
            )
            self._app.prepare(ctx_id=-1, det_size=(320, 320))
            print("✅ InsightFace detector ready")
        except Exception as e:
            self.logger.error(f"InsightFace init failed: {e}")
            self._app = None
            print(f"⚠️ Face detector unavailable: {e}")

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces, returns list of (x, y, w, h)"""
        if self._app is None:
            return []
        try:
            faces = self._app.get(image)
            results = []
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                w, h = x2 - x1, y2 - y1
                if w >= self.config.min_face_size[0] and h >= self.config.min_face_size[1]:
                    results.append((x1, y1, w, h))
            return results
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []

    def get_largest_face(self, faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """Get the largest detected face"""
        if not faces:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Kept for API compatibility"""
        return image
