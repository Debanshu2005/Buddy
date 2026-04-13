"""
Quick test for InsightFace and YOLOv8n
"""

from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parent / "models"

def test_insightface():
    print("Testing InsightFace...")
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_sc",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"]
        )
        app.prepare(ctx_id=-1, det_size=(320, 320))
        print("✅ InsightFace OK")
        return True
    except Exception as e:
        print(f"❌ InsightFace failed: {e}")
        return False


def test_yolo():
    print("Testing YOLOv8n...")
    try:
        from ultralytics import YOLO
        model = YOLO(str(MODELS_DIR / "yolov8n.pt"))
        print("✅ YOLOv8n OK")
        return True
    except Exception as e:
        print(f"❌ YOLOv8n failed: {e}")
        return False


def test_onnx():
    print("Testing ONNX Runtime...")
    try:
        import onnxruntime as ort
        print(f"✅ ONNX Runtime OK — version {ort.__version__}")
        return True
    except Exception as e:
        print(f"❌ ONNX Runtime failed: {e}")
        return False


if __name__ == "__main__":
    test_onnx()
    test_insightface()  # will download buffalo_sc model (~200MB) on first run
    test_yolo()         # will download yolov8n.pt (~6MB) on first run
    print("\nDone.")
