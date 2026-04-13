import onnxruntime as ort
import cv2
import numpy as np
import os
from pathlib import Path

# 80 COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


class ObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.session = None
        self.input_name = None
        self.input_size = (640, 640)
        self._init_model()

    def _init_model(self):
        """Download YOLOv8n ONNX if not present, then load it"""
        models_dir = Path(__file__).resolve().parent.parent / "models"
        model_path = models_dir / "yolov8n.onnx"

        if not os.path.exists(model_path):
            print("📥 YOLOv8n ONNX not found — downloading...")
            self._download_model(model_path)

        if not os.path.exists(model_path):
            print("❌ YOLOv8n model unavailable")
            return

        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            self.session = ort.InferenceSession(
                str(model_path), opts, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            print("✅ YOLOv8n ONNX loaded")
        except Exception as e:
            print(f"❌ Failed to load YOLOv8n: {e}")
            self.session = None

    def _download_model(self, path: Path):
        """Download YOLOv8n ONNX from ultralytics"""
        try:
            from ultralytics import YOLO
            model = YOLO(str(path.with_suffix(".pt")))
            model.export(format="onnx", imgsz=640, simplify=True)
            exported = "yolov8n.onnx"
            if os.path.exists(exported):
                os.makedirs(path.parent, exist_ok=True)
                os.rename(exported, path)
                print(f"✅ YOLOv8n exported to {path}")
        except Exception as e:
            print(f"❌ Model download/export failed: {e}")
            print("   Run: pip install ultralytics  then restart")

    def _preprocess(self, image: np.ndarray):
        """Letterbox resize to 640x640, normalize to float32"""
        h, w = image.shape[:2]
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (nw, nh))

        # Pad to 640x640
        canvas = np.full((self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8)
        pad_y = (self.input_size[0] - nh) // 2
        pad_x = (self.input_size[1] - nw) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        # HWC → CHW, normalize
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(blob, 0), scale, pad_x, pad_y

    def detect(self, frame: np.ndarray) -> list:
        """Run YOLOv8n inference, return list of detection dicts"""
        if self.session is None:
            return []
        try:
            h, w = frame.shape[:2]
            blob, scale, pad_x, pad_y = self._preprocess(frame)
            outputs = self.session.run(None, {self.input_name: blob})[0]

            # YOLOv8 output: [1, 84, 8400] → transpose to [8400, 84]
            preds = outputs[0].T  # (8400, 84)

            results = []
            for pred in preds:
                scores = pred[4:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence < self.confidence_threshold:
                    continue

                # cx, cy, w_box, h_box in 640-space
                cx, cy, bw, bh = pred[:4]

                # Remove padding and scale back to original image coords
                x1 = int((cx - bw / 2 - pad_x) / scale)
                y1 = int((cy - bh / 2 - pad_y) / scale)
                x2 = int((cx + bw / 2 - pad_x) / scale)
                y2 = int((cy + bh / 2 - pad_y) / scale)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"

                results.append({
                    "name": name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "area": (x2 - x1) * (y2 - y1)
                })

            # NMS to remove overlapping boxes
            return self._nms(results)

        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []

    def _nms(self, detections: list, iou_threshold: float = 0.45) -> list:
        """Simple NMS across all classes"""
        if not detections:
            return []

        detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        kept = []

        for det in detections:
            dominated = False
            for kept_det in kept:
                if self._iou(det["bbox"], kept_det["bbox"]) > iou_threshold:
                    dominated = True
                    break
            if not dominated:
                kept.append(det)

        return kept

    def _iou(self, a: list, b: list) -> float:
        """Intersection over Union"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            name = det["name"]
            conf = det["confidence"]

            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.5 else (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{name} {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        return frame
