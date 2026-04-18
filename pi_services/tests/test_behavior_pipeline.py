"""Basic regression tests for the lightweight behavior pipeline."""

from __future__ import annotations

import unittest

import numpy as np

from vision.behavior.classifier import BehaviorClassifier
from vision.behavior.feature_extractor import FeatureExtractor
from vision.behavior.pose_module import PoseFrame


def build_pose_frame(
    timestamp: float,
    torso_horizontal: bool = False,
    movement_offset: float = 0.0,
) -> PoseFrame:
    """Create a synthetic pose frame for deterministic unit tests."""
    keypoints = np.zeros((33, 3), dtype=np.float32)

    shoulders_y = 0.35 + movement_offset
    hips_y = 0.55 + movement_offset
    ankles_y = 0.82 + movement_offset
    left_shoulder = (0.42, shoulders_y)
    right_shoulder = (0.58, shoulders_y)
    left_hip = (0.44, hips_y)
    right_hip = (0.56, hips_y)
    left_ankle = (0.46, ankles_y)
    right_ankle = (0.54, ankles_y)

    if torso_horizontal:
        left_shoulder = (0.30, 0.58)
        right_shoulder = (0.52, 0.60)
        left_hip = (0.52, 0.59)
        right_hip = (0.71, 0.61)
        left_ankle = (0.78, 0.62)
        right_ankle = (0.90, 0.63)

    keypoints[0] = (0.5, 0.20 + movement_offset, 0.95)
    keypoints[11] = (*left_shoulder, 0.95)
    keypoints[12] = (*right_shoulder, 0.95)
    keypoints[23] = (*left_hip, 0.95)
    keypoints[24] = (*right_hip, 0.95)
    keypoints[27] = (*left_ankle, 0.95)
    keypoints[28] = (*right_ankle, 0.95)

    return PoseFrame(
        timestamp=timestamp,
        frame_size=(224, 224),
        person_bbox=(40, 20, 180, 210),
        keypoints=keypoints,
        visible_ratio=0.7,
        person_confidence=0.9,
        has_person=True,
    )


class FeatureExtractorTests(unittest.TestCase):
    def test_inactivity_increases_when_pose_is_static(self) -> None:
        extractor = FeatureExtractor(sequence_length=6)
        extractor.update(build_pose_frame(timestamp=1.0))
        features = extractor.update(build_pose_frame(timestamp=6.0))
        self.assertGreaterEqual(features.metrics["inactivity_seconds"], 4.9)

    def test_horizontal_pose_raises_orientation_score(self) -> None:
        extractor = FeatureExtractor(sequence_length=6)
        features = extractor.update(build_pose_frame(timestamp=1.0, torso_horizontal=True))
        self.assertGreater(features.metrics["body_orientation_score"], 0.8)


class BehaviorClassifierTests(unittest.TestCase):
    def test_heuristic_classifier_prefers_fall_for_drop_plus_horizontal_pose(self) -> None:
        extractor = FeatureExtractor(sequence_length=6)
        extractor.update(build_pose_frame(timestamp=1.0))
        feature_vector = extractor.update(build_pose_frame(timestamp=1.3, torso_horizontal=True, movement_offset=0.25))
        result = BehaviorClassifier().predict(feature_vector)
        self.assertGreater(result.probabilities["fall"], result.probabilities["normal"])


if __name__ == "__main__":
    unittest.main()
