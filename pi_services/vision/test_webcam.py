"""
Live webcam test - face detection + recognition + object detection
Press Q to quit
"""
import cv2
from core.config import Config
from vision.face_detector import FaceDetector
from vision.face_recognizer import FaceRecognizer
from vision.objrecog.obj import ObjectDetector

config = Config()
face_detector = FaceDetector(None, config)
face_recognizer = FaceRecognizer(None, config)
obj_detector = ObjectDetector(confidence_threshold=0.5)

cap = cv2.VideoCapture(0)
print("📷 Webcam started — press Q to quit")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Face detection
    faces = face_detector.detect(frame)
    for (x, y, w, h) in faces:
        color = (0, 255, 0)
        label = "Face"

        # Recognize every 10 frames to save CPU
        if frame_count % 10 == 0:
            face_crop = frame[y:y+h, x:x+w]
            name, confidence = face_recognizer.recognize(face_crop)
            if name != "Unknown":
                label = f"{name} ({confidence:.0%})"
                print(f"✅ Recognized: {name} ({confidence:.2f})")
            else:
                label = "Unknown"
                color = (0, 255, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Object detection
    detections = obj_detector.detect(frame)
    frame = obj_detector.draw_detections(frame, detections)

    if detections:
        names = [d['name'] for d in detections]
        print(f"🔍 Objects: {names}")

    # Status
    cv2.putText(frame, f"Faces: {len(faces)}  Objects: {len(detections)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Buddy Vision Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
