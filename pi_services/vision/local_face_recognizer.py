"""
Raspberry Pi-safe face recognizer that prefers repo-local ONNX models.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from memory.pi_memory import save_face, get_all_faces


LOCAL_MODEL_CANDIDATES = [
    Path(__file__).resolve().parent / "models" / "MobileFaceNet.onnx",
    Path(__file__).resolve().parent / "models" / "model.onnx",
]
LEGACY_MODEL_PATH = Path(os.path.expanduser("~")) / ".insightface" / "models" / "buffalo_sc" / "w600k_mbf.onnx"


class FaceRecognizer:
    def __init__(self, model_path: str, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.input_name = None
        self.model_path = self._resolve_model_path(model_path)
        self.known_faces: Dict[str, List[np.ndarray]] = {}
        self._init_model()
        self.load_known_faces()

    def _resolve_model_path(self, model_path: Optional[str]) -> Path:
        candidates = []
        if model_path:
            candidates.append(Path(model_path))
        candidates.extend(LOCAL_MODEL_CANDIDATES)
        candidates.append(LEGACY_MODEL_PATH)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _init_model(self):
        if not self.model_path.exists():
            print(f"⚠️ Recognition model not found at {self.model_path}")
            return
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            self.session = ort.InferenceSession(
                str(self.model_path), opts, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            print(f"✅ Face recognizer ready ({self.model_path.name})")
        except Exception as e:
            self.logger.error(f"Face recognizer init failed: {e}")
            self.session = None

    def _preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_img, (112, 112))
        # check what layout the model expects
        input_shape = self.session.get_inputs()[0].shape  # e.g. [1,3,112,112] or [1,112,112,3]
        if len(input_shape) == 4 and input_shape[1] == 3:
            # NCHW: (1, 3, 112, 112)
            blob = (resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) - 127.5) / 127.5
        else:
            # NHWC: (1, 112, 112, 3)
            blob = (resized[:, :, ::-1].astype(np.float32) - 127.5) / 127.5
        return np.expand_dims(blob, 0)

    def get_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        if self.session is None:
            return None
        try:
            blob = self._preprocess_face(face_img)
            emb = self.session.run(None, {self.input_name: blob})[0].flatten()
            norm = np.linalg.norm(emb)
            return emb / (norm + 1e-6)
        except Exception as e:
            self.logger.error(f"Embedding error: {e}")
            return None

    def recognize(self, face_img: np.ndarray) -> Tuple[str, float]:
        if self.session is None:
            return "Unknown", 0.0
        try:
            embedding = self.get_embedding(face_img)
            if embedding is None or not self.known_faces:
                return "Unknown", 0.0

            best_match = None
            best_score = -1.0
            for name, embeddings in self.known_faces.items():
                for known_emb in embeddings:
                    score = float(np.dot(embedding, known_emb))
                    if score > best_score:
                        best_score = score
                        best_match = name

            if best_score >= 0.35:
                return best_match, best_score
            return "Unknown", 0.0
        except Exception as e:
            self.logger.error(f"Recognition error: {e}")
            return "Unknown", 0.0

    def add_face(self, name: str, face_img: np.ndarray, angle: str = "front") -> bool:
        try:
            embedding = self.get_embedding(face_img)
            if embedding is None:
                print(f"⚠️ Could not extract embedding for {name}/{angle}")
                return False

            if name not in self.known_faces:
                self.known_faces[name] = []
            self.known_faces[name].append(embedding)
            save_face(name, embedding, angle)
            print(f"✅ Face registered for {name} ({angle})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add face for {name}: {e}")
            return False

    def load_known_faces(self) -> None:
        try:
            faces = get_all_faces()
            self.known_faces = {}
            for name, embeddings in faces.items():
                normalized = []
                for e in embeddings:
                    emb = np.array(e, dtype=np.float32).flatten()
                    norm = np.linalg.norm(emb)
                    normalized.append(emb / (norm + 1e-6))
                self.known_faces[name] = normalized
            total = sum(len(v) for v in self.known_faces.values())
            print(f"✅ Loaded {len(self.known_faces)} people ({total} embeddings): {list(self.known_faces.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to load known faces: {e}")
            self.known_faces = {}
