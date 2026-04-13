"""
Face Recognition Module
Uses InsightFace buffalo_sc w600k_mbf.onnx directly via ONNX Runtime
Supports multiple embeddings per person (different angles)
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
import logging
from typing import Dict, Tuple, Optional, List
from pi_memory import save_face, get_all_faces

MODEL_PATH = os.path.join(
    os.path.expanduser("~"), ".insightface", "models", "buffalo_sc", "w600k_mbf.onnx"
)


class FaceRecognizer:

    def __init__(self, model_path: str, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.known_faces: Dict[str, List[np.ndarray]] = {}  # {name: [emb1, emb2, ...]}
        self._init_model()
        self.load_known_faces()

    def _init_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️ Recognition model not found at {MODEL_PATH}")
            return
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            self.session = ort.InferenceSession(
                MODEL_PATH, opts, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            print("✅ Face recognizer ready (w600k_mbf ArcFace ONNX)")
        except Exception as e:
            self.logger.error(f"Face recognizer init failed: {e}")
            self.session = None

    def _preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_img, (112, 112))
        blob = (resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) - 127.5) / 127.5
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
        """Compare against all stored angles, return best match"""
        if self.session is None:
            return "Unknown", 0.0
        try:
            embedding = self.get_embedding(face_img)
            if embedding is None:
                return "Unknown", 0.0

            if not self.known_faces:
                return "Unknown", 0.0

            best_match = None
            best_score = -1.0

            for name, embeddings in self.known_faces.items():
                for known_emb in embeddings:
                    score = float(np.dot(embedding, known_emb))
                    if score > best_score:
                        best_score = score
                        best_match = name

            if best_score >= 0.45:
                return best_match, best_score

            return "Unknown", 0.0

        except Exception as e:
            self.logger.error(f"Recognition error: {e}")
            return "Unknown", 0.0

    def add_face(self, name: str, face_img: np.ndarray, angle: str = "front") -> bool:
        """Add a face embedding for a specific angle"""
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
        """Load all face embeddings from DB — supports multiple angles per person"""
        try:
            faces = get_all_faces()  # {name: [emb1, emb2, ...]}
            self.known_faces = {
                name: [np.array(e, dtype=np.float32).flatten() for e in embeddings]
                for name, embeddings in faces.items()
            }
            total = sum(len(v) for v in self.known_faces.values())
            print(f"✅ Loaded {len(self.known_faces)} people ({total} embeddings): {list(self.known_faces.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to load known faces: {e}")
            self.known_faces = {}
