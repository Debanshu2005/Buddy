"""Lightweight classifier for behavior-risk scoring on edge devices."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import pickle
from typing import Any, Optional

import numpy as np

from .feature_extractor import FeatureVector


@dataclass
class ClassificationResult:
    """Probabilistic risk estimates plus classifier metadata."""

    probabilities: dict[str, float]
    source: str


class BehaviorClassifier:
    """Use a small sklearn model when available, otherwise a calibrated heuristic model."""

    LABELS = ("fall", "injury", "no_movement", "normal")

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self._model: Optional[Any] = None
        self._load_model()

    def predict(self, feature_vector: FeatureVector) -> ClassificationResult:
        """Return class probabilities for the current temporal feature vector."""
        if self._model is not None:
            probabilities = self._predict_with_model(feature_vector.values)
            return ClassificationResult(probabilities=probabilities, source="model")
        probabilities = self._predict_with_heuristics(feature_vector.metrics)
        return ClassificationResult(probabilities=probabilities, source="heuristic")

    def _load_model(self) -> None:
        if not self.model_path or not self.model_path.exists():
            return
        with self.model_path.open("rb") as handle:
            self._model = pickle.load(handle)

    def _predict_with_model(self, values: np.ndarray) -> dict[str, float]:
        batch = values.reshape(1, -1)
        if hasattr(self._model, "predict_proba"):
            model_probs = self._model.predict_proba(batch)
            if isinstance(model_probs, list):
                raw = {label: float(proba[0][1]) for label, proba in zip(self.LABELS[:-1], model_probs)}
            else:
                classes = getattr(self._model, "classes_", list(self.LABELS))
                raw = {str(label): float(prob) for label, prob in zip(classes, model_probs[0])}
        else:
            raw_scores = np.asarray(self._model.predict(batch)[0], dtype=np.float32)
            raw_scores = self._softmax(raw_scores)
            raw = {label: float(prob) for label, prob in zip(self.LABELS, raw_scores)}
        return self._normalize(raw)

    def _predict_with_heuristics(self, metrics: dict[str, float]) -> dict[str, float]:
        horizontal = np.clip((metrics["body_orientation_score"] - 0.65) / 1.2, 0.0, 1.0)
        low_torso = np.clip((0.18 - metrics["torso_height_ratio"]) / 0.18, 0.0, 1.0)
        fast_drop = np.clip(metrics["sudden_drop_score"] / 1.8, 0.0, 1.0)
        inactivity = np.clip(metrics["inactivity_seconds"] / 12.0, 0.0, 1.0)
        low_motion = np.clip((0.025 - metrics["centroid_speed"]) / 0.025, 0.0, 1.0)
        posture_abnormal = np.clip(horizontal * 0.7 + low_torso * 0.3, 0.0, 1.0)

        fall = self._sigmoid(3.8 * fast_drop + 2.4 * posture_abnormal - 2.6)
        injury = self._sigmoid(2.6 * posture_abnormal + 1.4 * inactivity + 1.0 * low_motion - 2.3)
        no_movement = self._sigmoid(3.2 * inactivity + 1.1 * low_motion - 2.0)
        normal = self._sigmoid(
            2.8 * metrics["has_person"]
            + 1.4 * metrics["visible_ratio"]
            - 2.0 * fall
            - 1.6 * injury
            - 1.8 * no_movement
            - 1.3
        )
        return self._normalize(
            {
                "fall": fall,
                "injury": injury,
                "no_movement": no_movement,
                "normal": normal,
            }
        )

    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        normalized = {label: max(0.0, float(scores.get(label, 0.0))) for label in BehaviorClassifier.LABELS}
        total = sum(normalized.values())
        if total <= 0:
            return {label: 0.0 for label in BehaviorClassifier.LABELS}
        return {label: value / total for label, value in normalized.items()}

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))
