"""
Face Registration Script
Run this to register your face into the database
"""
import cv2
import time
from core.config import Config
from vision.face_recognizer import FaceRecognizer

config = Config()
recognizer = FaceRecognizer(None, config)

name = input("Enter your name: ").strip()
if not name:
    print("No name entered, exiting.")
    exit()

cap = cv2.VideoCapture(0)
print(f"\nRegistering face for: {name}")
print("Look at the camera — capturing in 3 seconds...")
time.sleep(3)

saved = False
attempts = 0

while not saved and attempts < 30:
    ret, frame = cap.read()
    if not ret:
        continue

    attempts += 1
    result = recognizer.add_face(name, frame)

    if result:
        print(f"✅ Face registered for {name}!")
        saved = True
    else:
        print(f"⚠️ Could not detect face clearly, retrying... ({attempts}/30)")
        time.sleep(0.3)

cap.release()

if not saved:
    print("❌ Failed to register face. Make sure your face is clearly visible and well lit.")
else:
    print(f"\n✅ Done! {name} is now registered.")
    print("Buddy will now recognize you when you run Buddy_new.py")
