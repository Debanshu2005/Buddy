"""Temporal feature engineering for lightweight behavior detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Optional

import numpy as np

from .pose_module import PoseFrame


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
NOSE = 0


@dataclass
class FeatureVector:
    """Feature payload passed into the classifier and API response."""

    values: np.ndarray
    names: list[str]
    metrics: dict[str, float]


class FeatureExtractor:
    """Build robust temporal features from pose landmarks over recent frames."""

    def __init__(self, sequence_length: int = 12, inactivity_velocity_threshold: float = 0.012) -> None:
        self.sequence_length = sequence_length
        self.inactivity_velocity_threshold = inactivity_velocity_threshold
        self._history: Deque[PoseFrame] = deque(maxlen=sequence_length)
        self._last_active_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Clear temporal state for a new stream/session."""
        self._history.clear()
        self._last_active_timestamp = None

    def update(self, pose_frame: PoseFrame) -> FeatureVector:
        """Append pose state and compute temporal summary features."""
        self._history.append(pose_frame)
        metrics = self._compute_metrics()
        values = np.array([metrics[name] for name in self.feature_names()], dtype=np.float32)
        return FeatureVector(values=values, names=self.feature_names(), metrics=metrics)

    @staticmethod
    def feature_names() -> list[str]:
        """Return the stable feature ordering used for inference."""
        return [
            "has_person",
            "visible_ratio",
            "torso_angle_deg",
            "body_orientation_score",
            "torso_height_ratio",
            "hip_to_ankle_ratio",
            "centroid_speed",
            "head_drop_speed",
            "sudden_drop_score",
            "movement_variance",
            "inactivity_seconds",
            "bbox_area_ratio",
            "pose_confidence",
        ]

    def _compute_metrics(self) -> dict[str, float]:
        latest = self._history[-1]
        keypoints = latest.keypoints
        width, height = latest.frame_size

        shoulder_mid = self._midpoint(keypoints, LEFT_SHOULDER, RIGHT_SHOULDER)
        hip_mid = self._midpoint(keypoints, LEFT_HIP, RIGHT_HIP)
        ankle_mid = self._midpoint(keypoints, LEFT_ANKLE, RIGHT_ANKLE)
        centroid = self._visible_centroid(keypoints)

        torso_vector = shoulder_mid - hip_mid
        torso_angle_deg = abs(math.degrees(math.atan2(torso_vector[1], torso_vector[0]))) if self._is_valid_vector(torso_vector) else 90.0
        body_orientation_score = float(np.clip(abs(torso_vector[0]) / (abs(torso_vector[1]) + 1e-6), 0.0, 2.5)) if self._is_valid_vector(torso_vector) else 0.0
        torso_height_ratio = float(np.clip(abs(shoulder_mid[1] - hip_mid[1]), 0.0, 1.0))
        hip_to_ankle_ratio = float(np.clip(abs(hip_mid[1] - ankle_mid[1]), 0.0, 1.0))
        bbox_area_ratio = self._bbox_area_ratio(latest.person_bbox, width, height)
        pose_confidence = float(latest.visible_ratio * latest.person_confidence)

        centroid_speed = 0.0
        head_drop_speed = 0.0
        movement_variance = 0.0
        sudden_drop_score = 0.0
        inactivity_seconds = 0.0

        if len(self._history) >= 2:
            previous = self._history[-2]
            dt = max(latest.timestamp - previous.timestamp, 1e-3)
            previous_centroid = self._visible_centroid(previous.keypoints)
            centroid_speed = float(np.linalg.norm(centroid - previous_centroid) / dt)
            head_drop_speed = float(abs(keypoints[NOSE, 1] - previous.keypoints[NOSE, 1]) / dt)
            sudden_drop_score = float(np.clip(head_drop_speed * (1.0 + body_orientation_score), 0.0, 5.0))

        if len(self._history) >= 3:
            centroids = np.array([self._visible_centroid(frame.keypoints) for frame in self._history], dtype=np.float32)
            diffs = np.diff(centroids, axis=0)
            movement_variance = float(np.var(np.linalg.norm(diffs, axis=1))) if len(diffs) else 0.0

        motion_level = max(centroid_speed, head_drop_speed)
        if motion_level > self.inactivity_velocity_threshold:
            self._last_active_timestamp = latest.timestamp
        elif self._last_active_timestamp is None:
            self._last_active_timestamp = latest.timestamp

        inactivity_seconds = max(0.0, latest.timestamp - (self._last_active_timestamp or latest.timestamp))

        return {
            "has_person": 1.0 if latest.has_person else 0.0,
            "visible_ratio": float(latest.visible_ratio),
            "torso_angle_deg": float(np.clip(torso_angle_deg, 0.0, 180.0)),
            "body_orientation_score": body_orientation_score,
            "torso_height_ratio": torso_height_ratio,
            "hip_to_ankle_ratio": hip_to_ankle_ratio,
            "centroid_speed": centroid_speed,
            "head_drop_speed": head_drop_speed,
            "sudden_drop_score": sudden_drop_score,
            "movement_variance": movement_variance,
            "inactivity_seconds": inactivity_seconds,
            "bbox_area_ratio": bbox_area_ratio,
            "pose_confidence": pose_confidence,
        }

    @staticmethod
    def _midpoint(keypoints: np.ndarray, idx_a: int, idx_b: int) -> np.ndarray:
        point_a = keypoints[idx_a, :2]
        point_b = keypoints[idx_b, :2]
        return (point_a + point_b) / 2.0

    @staticmethod
    def _visible_centroid(keypoints: np.ndarray) -> np.ndarray:
        mask = keypoints[:, 2] > 0.4
        if not np.any(mask):
            return np.array([0.5, 0.5], dtype=np.float32)
        return np.mean(keypoints[mask, :2], axis=0)

    @staticmethod
    def _bbox_area_ratio(
        bbox: Optional[tuple[int, int, int, int]],
        frame_width: int,
        frame_height: int,
    ) -> float:
        if bbox is None or frame_width <= 0 or frame_height <= 0:
            return 0.0
        x1, y1, x2, y2 = bbox
        return float(((x2 - x1) * (y2 - y1)) / max(frame_width * frame_height, 1))

    @staticmethod
    def _is_valid_vector(vector: np.ndarray) -> bool:
        return bool(np.isfinite(vector).all() and np.linalg.norm(vector) > 0.0)
