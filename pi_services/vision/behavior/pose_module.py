"""Pose extraction helpers for lightweight behavior analysis."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - optional dependency in dev environments
    mp = None

try:
    from vision.objrecog.obj import ObjectDetector
except Exception:  # pragma: no cover - optional dependency in dev environments
    ObjectDetector = None


LOGGER = logging.getLogger(__name__)


@dataclass
class PoseFrame:
    """Single-frame pose payload used by the temporal pipeline."""

    timestamp: float
    frame_size: tuple[int, int]
    person_bbox: Optional[tuple[int, int, int, int]]
    keypoints: np.ndarray
    visible_ratio: float
    person_confidence: float
    has_person: bool


class PoseExtractor:
    """Detect a person and extract pose landmarks with MediaPipe Pose."""

    def __init__(
        self,
        model_complexity: int = 0,
        enable_person_detector: bool = True,
        min_detection_confidence: float = 0.45,
        min_tracking_confidence: float = 0.45,
        person_confidence_threshold: float = 0.35,
    ) -> None:
        self.person_confidence_threshold = person_confidence_threshold
        self._mp_pose: Optional[Any] = None
        self._pose: Optional[Any] = None
        self._person_detector = None

        if mp is not None:
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            LOGGER.warning("MediaPipe not available; pose extraction will return empty landmarks.")

        if enable_person_detector and ObjectDetector is not None:
            try:
                self._person_detector = ObjectDetector(confidence_threshold=person_confidence_threshold)
            except Exception as exc:  # pragma: no cover - hardware specific
                LOGGER.warning("Person detector unavailable: %s", exc)

    def extract(self, frame: np.ndarray, timestamp: float) -> PoseFrame:
        """Return person-localized pose landmarks for a single RGB/BGR frame."""
        height, width = frame.shape[:2]
        bbox, confidence = self._detect_person(frame)
        crop = self._crop_person(frame, bbox) if bbox is not None else frame

        keypoints = np.zeros((33, 3), dtype=np.float32)
        visible_ratio = 0.0

        if self._pose is not None:
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = self._pose.process(rgb_crop)
            if result.pose_landmarks:
                keypoints = self._landmarks_to_keypoints(
                    result.pose_landmarks.landmark,
                    bbox=bbox,
                    frame_width=width,
                    frame_height=height,
                    crop_shape=crop.shape[:2],
                )
                visible_ratio = float(np.mean(keypoints[:, 2] > 0.4))

        has_person = bbox is not None and confidence >= self.person_confidence_threshold
        if visible_ratio > 0.15:
            has_person = True

        return PoseFrame(
            timestamp=timestamp,
            frame_size=(width, height),
            person_bbox=bbox,
            keypoints=keypoints,
            visible_ratio=visible_ratio,
            person_confidence=confidence,
            has_person=has_person,
        )

    def draw_pose(self, frame: np.ndarray, pose_frame: PoseFrame) -> np.ndarray:
        """Overlay a bounding box and pose skeleton for monitoring/debugging."""
        output = frame.copy()
        if pose_frame.person_bbox:
            x1, y1, x2, y2 = pose_frame.person_bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 180, 255), 2)

        for x_norm, y_norm, visibility in pose_frame.keypoints:
            if visibility < 0.4:
                continue
            x_px = int(x_norm * pose_frame.frame_size[0])
            y_px = int(y_norm * pose_frame.frame_size[1])
            cv2.circle(output, (x_px, y_px), 3, (0, 255, 0), -1)

        for start_idx, end_idx in self._connection_pairs():
            if pose_frame.keypoints[start_idx, 2] < 0.4 or pose_frame.keypoints[end_idx, 2] < 0.4:
                continue
            start = (
                int(pose_frame.keypoints[start_idx, 0] * pose_frame.frame_size[0]),
                int(pose_frame.keypoints[start_idx, 1] * pose_frame.frame_size[1]),
            )
            end = (
                int(pose_frame.keypoints[end_idx, 0] * pose_frame.frame_size[0]),
                int(pose_frame.keypoints[end_idx, 1] * pose_frame.frame_size[1]),
            )
            cv2.line(output, start, end, (255, 180, 0), 2)

        return output

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._pose is not None:
            self._pose.close()

    def _detect_person(self, frame: np.ndarray) -> tuple[Optional[tuple[int, int, int, int]], float]:
        if self._person_detector is None:
            return None, 0.0

        detections = self._person_detector.detect(frame)
        persons = [d for d in detections if d.get("name") == "person"]
        if not persons:
            return None, 0.0

        persons.sort(key=lambda item: (item.get("confidence", 0.0), item.get("area", 0)), reverse=True)
        x1, y1, x2, y2 = persons[0]["bbox"]
        return (x1, y1, x2, y2), float(persons[0].get("confidence", 0.0))

    @staticmethod
    def _crop_person(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _landmarks_to_keypoints(
        landmarks: Any,
        bbox: Optional[tuple[int, int, int, int]],
        frame_width: int,
        frame_height: int,
        crop_shape: tuple[int, int],
    ) -> np.ndarray:
        keypoints = np.zeros((33, 3), dtype=np.float32)
        crop_height, crop_width = crop_shape
        bbox_x1, bbox_y1 = (bbox[0], bbox[1]) if bbox else (0, 0)

        for idx, landmark in enumerate(landmarks):
            x_px = landmark.x * crop_width + bbox_x1
            y_px = landmark.y * crop_height + bbox_y1
            keypoints[idx, 0] = np.clip(x_px / max(frame_width, 1), 0.0, 1.0)
            keypoints[idx, 1] = np.clip(y_px / max(frame_height, 1), 0.0, 1.0)
            keypoints[idx, 2] = np.clip(float(getattr(landmark, "visibility", 0.0)), 0.0, 1.0)

        return keypoints

    def _connection_pairs(self) -> list[tuple[int, int]]:
        if self._mp_pose is None:
            return []
        return [(start.value, end.value) for start, end in self._mp_pose.POSE_CONNECTIONS]
