"""End-to-end pipeline for real-time behavior risk detection."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

from .classifier import BehaviorClassifier
from .feature_extractor import FeatureExtractor
from .pose_module import PoseExtractor


LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Operational settings for Raspberry Pi behavior monitoring."""

    resize_width: int = 224
    resize_height: int = 224
    process_every_n_frames: int = 2
    sequence_length: int = 12
    camera_index: int = 0
    target_fps: int = 6
    classifier_model_path: Optional[str] = None
    alert_webhook_url: Optional[str] = None
    enable_visualization: bool = False


class BehaviorDetectionPipeline:
    """Coordinate frame processing, temporal modeling, and alert decisions."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        alert_callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.pose_extractor = PoseExtractor()
        self.feature_extractor = FeatureExtractor(sequence_length=self.config.sequence_length)
        self.classifier = BehaviorClassifier(model_path=self.config.classifier_model_path)
        self.alert_callback = alert_callback or self._default_alert_callback
        self._frame_counter = 0
        self._lock = threading.Lock()
        self._last_result: Optional[dict] = None

    def reset(self) -> None:
        """Reset temporal state for a new video stream."""
        with self._lock:
            self._frame_counter = 0
            self._last_result = None
            self.feature_extractor.reset()

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> dict:
        """Process a single frame and return probabilistic behavior-risk output."""
        with self._lock:
            timestamp = float(timestamp or time.time())
            self._frame_counter += 1

            resized = cv2.resize(frame, (self.config.resize_width, self.config.resize_height))
            if self._frame_counter % max(self.config.process_every_n_frames, 1) != 0 and self._last_result is not None:
                return {
                    **self._last_result,
                    "frame_processed": False,
                    "timestamp": timestamp,
                }

            pose_frame = self.pose_extractor.extract(resized, timestamp=timestamp)
            feature_vector = self.feature_extractor.update(pose_frame)
            classification = self.classifier.predict(feature_vector)
            decision = self._refine_decision(classification.probabilities, feature_vector.metrics)

            visualization = None
            if self.config.enable_visualization:
                overlay = self.pose_extractor.draw_pose(resized, pose_frame)
                visualization = self._encode_jpeg(overlay)

            result = {
                "timestamp": timestamp,
                "frame_processed": True,
                "frame_size": {"width": self.config.resize_width, "height": self.config.resize_height},
                "person_detected": pose_frame.has_person,
                "person_confidence": round(pose_frame.person_confidence, 4),
                "pose_visibility": round(pose_frame.visible_ratio, 4),
                "features": {key: round(float(value), 4) for key, value in feature_vector.metrics.items()},
                "probabilities": {key: round(float(value), 4) for key, value in classification.probabilities.items()},
                "decision": decision,
                "classifier_source": classification.source,
                "visualization_jpeg_base64": visualization,
            }
            self._last_result = result

            if decision["alert"]:
                self.alert_callback(result)

            return result

    def process_camera_stream(self, duration_seconds: Optional[float] = None) -> list[dict]:
        """Capture frames from a camera and process them at a Pi-safe cadence."""
        cap = cv2.VideoCapture(self.config.camera_index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        interval = 1.0 / max(self.config.target_fps, 1)
        results: list[dict] = []
        start = time.time()

        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                results.append(self.process_frame(frame))
                if duration_seconds and (time.time() - start) >= duration_seconds:
                    break
                time.sleep(interval)
        finally:
            cap.release()

        return results

    def _refine_decision(self, probabilities: dict[str, float], metrics: dict[str, float]) -> dict:
        label = max(probabilities, key=probabilities.get)
        alert = False
        severity = "low"
        reason = "normal activity profile"

        if probabilities["fall"] > 0.45 and probabilities["no_movement"] > 0.30:
            label = "possible_medical_emergency"
            alert = True
            severity = "critical"
            reason = "fall pattern followed by prolonged low movement"
        elif probabilities["injury"] > 0.35 and metrics["centroid_speed"] < 0.02 and metrics["body_orientation_score"] > 0.8:
            label = "injury_suspected"
            alert = True
            severity = "high"
            reason = "abnormal posture with sustained slow movement"
        elif probabilities["fall"] > 0.45:
            label = "fall_detected"
            alert = True
            severity = "high"
            reason = "sudden vertical drop and horizontal posture change"
        elif probabilities["no_movement"] > 0.40:
            label = "monitor_closely"
            severity = "medium"
            reason = "extended inactivity detected"

        return {
            "label": label,
            "alert": alert,
            "severity": severity,
            "reason": reason,
        }

    @staticmethod
    def _default_alert_callback(result: dict) -> None:
        LOGGER.warning("Behavior alert triggered: %s", result["decision"])

    @staticmethod
    def _encode_jpeg(frame: np.ndarray) -> Optional[str]:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None
        import base64

        return base64.b64encode(encoded.tobytes()).decode("ascii")
