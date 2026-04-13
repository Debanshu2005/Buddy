"""
Buddy Windows Test
- Face detection + recognition
- Voice output (Edge TTS via Windows)
- LLM conversation
- Multi-angle face registration
Press Q on the camera window to quit
"""
import cv2
import asyncio
import threading
import time
import os
import requests
import numpy as np
import sounddevice as sd
import pygame
from core.config import Config
from vision.face_detector import FaceDetector
from vision.face_recognizer import FaceRecognizer
from core.stability_tracker import StabilityTracker
from vision.objrecog.obj import ObjectDetector

# ── Config ────────────────────────────────────────────────────────────────────
LLM_URL   = "http://localhost:11434"
TTS_VOICE = "en-IN-NeerjaNeural"
# ─────────────────────────────────────────────────────────────────────────────

config    = Config()
detector  = FaceDetector(None, config)
recognizer = FaceRecognizer(None, config)
stability = StabilityTracker(config)
obj_det   = ObjectDetector(confidence_threshold=0.5)

# Init pygame once at startup
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Shared state
is_speaking     = False
active_user     = None
awaiting_name   = False
last_frame      = None
running         = True
current_objects = []
_objects_lock   = threading.Lock()

# Pre-load Whisper at startup so first response isn't slow
print("Loading Whisper model...")
from faster_whisper import WhisperModel
_whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
print("✅ Whisper ready")


# ── TTS ───────────────────────────────────────────────────────────────────────
async def _tts(text: str):
    import edge_tts, tempfile
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    tmp = tempfile.mktemp(suffix=".mp3")
    await communicate.save(tmp)
    pygame.mixer.music.load(tmp)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)
    pygame.mixer.music.unload()
    try:
        os.remove(tmp)
    except:
        pass


def speak(text: str):
    global is_speaking
    print(f"\n🤖 Buddy: {text}")

    def _run():
        global is_speaking
        is_speaking = True
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_tts(text))
        finally:
            loop.close()
        is_speaking = False

    threading.Thread(target=_run, daemon=True).start()


# ── STT ───────────────────────────────────────────────────────────────────────
def listen_windows() -> str:
    try:
        print("🎤 Listening...")
        audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        if float(np.sqrt(np.mean(audio ** 2))) < 0.01:
            print("🔇 No speech detected")
            return ""

        segments, _ = _whisper_model.transcribe(audio, language="en")
        text = " ".join(s.text for s in segments).strip()

        if text:
            print(f"🎤 You said: '{text}'")
            return text
        print("🔇 Could not understand")
        return ""
    except Exception as e:
        print(f"❌ STT error: {e}")
        return ""


# ── LLM ───────────────────────────────────────────────────────────────────────
def call_brain(user_input: str, recognized_user: str = None, objects: list = None) -> str:
    try:
        context = "You are Buddy, a friendly robot assistant. Keep replies short and conversational.\n"
        if recognized_user:
            context += f"You are talking to {recognized_user}.\n"
        if objects:
            context += f"You can currently see these objects through your camera: {', '.join(objects)}.\n"
        context += "Use the camera information to answer questions about what you see."

        resp = requests.post(
            f"{LLM_URL}/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": f"{context}\n\nUser: {user_input}\nBuddy:",
                "stream": False
            },
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"❌ LLM error: {e}")
    return "Sorry, I'm having trouble thinking right now."


# ── Multi-angle face registration ─────────────────────────────────────────────
def _register_multi_angle(name: str):
    poses = [
        ("front", "Great! Now slowly turn your face to the LEFT and hold it there."),
        ("left",  "Perfect! Now turn your face to the RIGHT and hold it there."),
        ("right", "Good! Now tilt your face slightly UP and hold it there."),
        ("up",    "Almost done! Now tilt your face slightly DOWN and hold it there."),
        ("down",  None),
    ]

    for angle, next_instruction in poses:
        time.sleep(4.0)
        captured = False
        for _ in range(30):
            if last_frame is None:
                time.sleep(0.1)
                continue
            faces = detector.detect(last_frame)
            if faces:
                x, y, w, h = detector.get_largest_face(faces)
                if w > 80 and h > 80:
                    face_roi = last_frame[y:y+h, x:x+w]
                    if recognizer.add_face(name, face_roi, angle):
                        print(f"✅ Captured {angle} for {name}")
                        captured = True
                        break
            time.sleep(0.1)

        if not captured:
            print(f"⚠️ Could not capture {angle} for {name}")
        if next_instruction:
            speak(next_instruction)

    print(f"✅ Multi-angle registration complete for {name}")


# ── Name extraction ───────────────────────────────────────────────────────────
# Common words that are not names
_NOT_NAMES = {
    'here', 'there', 'not', 'just', 'also', 'very', 'really', 'sorry',
    'good', 'fine', 'okay', 'ok', 'yes', 'no', 'sure', 'well', 'now',
    'still', 'already', 'going', 'trying', 'doing', 'being', 'having',
    'little', 'bit', 'back', 'home', 'ready', 'happy', 'busy', 'tired'
}

def extract_name(text: str) -> str:
    t = text.lower().strip()
    for phrase in ['my name is', 'i am', "i'm", 'call me', 'name is']:
        if phrase in t:
            after = t.split(phrase)[1].strip()
            for word in after.split():
                clean = ''.join(c for c in word if c.isalpha())
                if len(clean) > 1 and clean not in _NOT_NAMES:
                    return clean.title()
    return ""


# ── Conversation thread ───────────────────────────────────────────────────────
def conversation_loop():
    global active_user, awaiting_name, is_speaking, current_objects

    while running:
        while is_speaking:
            time.sleep(0.1)
        time.sleep(0.3)

        text = listen_windows()
        if not text:
            continue

        t_lower = text.lower()

        # Name introduction → multi-angle registration
        if any(p in t_lower for p in ['my name is', 'i am', "i'm", 'call me', 'name is']):
            name = extract_name(text)
            print(f"🔍 Extracted name: '{name}'")
            if name and len(name) > 1:
                active_user = name
                awaiting_name = False
                speak(f"Nice to meet you {name}! I'll scan your face from different angles. Please look straight at me and hold still.")
                time.sleep(5)
                _register_multi_angle(name)
                reply = call_brain(
                    f"You just registered {name}'s face from multiple angles. Greet them warmly.",
                    recognized_user=name,
                    objects=list(current_objects)
                )
                speak(reply)
                continue

        # Normal conversation
        with _objects_lock:
            objs = list(current_objects)
        print(f"📦 Passing to LLM: {objs}")
        reply = call_brain(text, active_user, objects=objs)
        speak(reply)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global active_user, awaiting_name, last_frame, running

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("📷 Camera started")

    frame_count    = 0
    last_recog_time = 0
    greeted        = False
    detections     = []
    last_faces     = []  # reuse last detection result for drawing

    threading.Thread(target=conversation_loop, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        last_frame = frame.copy()
        frame_count += 1

        # Object detection every 20 frames
        if frame_count % 20 == 0:
            detections = obj_det.detect(frame)
            new_objects = [d['name'] for d in detections if d['name'] != 'person']
            with _objects_lock:
                current_objects.clear()
                current_objects.extend(new_objects)
            if current_objects:
                print(f"👁️ Objects: {current_objects}")

        # Face detection + recognition every 15 frames
        if frame_count % 15 == 0:
            last_faces = detector.detect(frame)

            if last_faces:
                largest = detector.get_largest_face(last_faces)
                is_stable = stability.update(largest)
                x, y, w, h = largest

                if is_stable and w > 80 and h > 80:
                    now = time.time()
                    if now - last_recog_time > 5.0:
                        face_roi = frame[y:y+h, x:x+w]
                        name, conf = recognizer.recognize(face_roi)
                        last_recog_time = now

                        if name != "Unknown" and conf > 0.45:
                            if active_user != name:
                                active_user = name
                                print(f"✅ Recognized: {name} ({conf:.2f})")
                                if not greeted:
                                    greeted = True
                                    reply = call_brain(
                                        f"You just recognized {name}. Greet them warmly by name.",
                                        recognized_user=name
                                    )
                                    speak(reply)
                        else:
                            if not greeted and not awaiting_name and not is_speaking:
                                greeted = True
                                awaiting_name = True
                                reply = call_brain(
                                    "You see someone but don't recognize them. Say hi and ask for their name.",
                                    recognized_user=None
                                )
                                speak(reply)
            else:
                stability.reset()

        # Draw — reuse last_faces, no extra detection call
        for (x, y, w, h) in last_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if active_user:
                cv2.putText(frame, active_user, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if detections:
            frame = obj_det.draw_detections(frame, detections)

        status = f"Chatting with: {active_user}" if active_user else "Looking for people..."
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Buddy Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
