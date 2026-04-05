"""
Buddy Pi - Hardware Only Service
Face recognition, voice I/O, camera, objects - NO BRAIN
Optimized for Raspberry Pi 4B with Python 3.13
Uses WebSocket STT for speech recognition
"""
import subprocess
import shutil
import atexit
import cv2
import numpy as np
import time
import logging
import requests
import threading
import signal
import sys
from pathlib import Path
from typing import Optional
from collections import defaultdict

# Serial port for Arduino motor controller
ARDUINO_PORT = "/dev/ttyUSB0"   # change to /dev/ttyACM0 if needed
ARDUINO_BAUD = 115200
MOTOR_MOVE_DURATION = 2.0        # seconds per voice-commanded move

# Hardware modules only
from config import Config
from states import BuddyState, StateManager
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from stability_tracker import StabilityTracker
from objrecog.obj import ObjectDetector
from servo_controller import ServoController
from motor_controller import MotorController

# ── WebSocket STT (replaces Whisper STT) ────────────────────────────────────
import asyncio
import sounddevice as sd
from scipy.signal import resample_poly

STT_SERVER_IP  = "192.168.31.59"
STT_PORT       = 8765
MIC_RATE       = 48000   # Native mic sample rate
TARGET_RATE    = 16000   # Rate the STT server expects
CHUNK_DURATION = 3       # Seconds of audio per request

try:
    import websockets as _websockets_lib
    _websockets_available = True
except ImportError as e:
    print(f"❌ Failed to import websockets: {e}")
    _websockets_available = False


async def _ws_listen_once() -> str:
    """
    Record one audio chunk, send it to the WebSocket STT server,
    and return the transcribed text (empty string on failure).
    """
    uri = f"ws://{STT_SERVER_IP}:{STT_PORT}"
    try:
        async with _websockets_lib.connect(uri) as websocket:
            audio = sd.rec(
                int(CHUNK_DURATION * MIC_RATE),
                samplerate=MIC_RATE,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            audio = np.squeeze(audio)

            # Resample 48 kHz → 16 kHz
            audio_16k = resample_poly(audio, TARGET_RATE, MIC_RATE)
            await websocket.send(audio_16k.astype(np.float32).tobytes())

            text = await websocket.recv()
            return text.strip() if text else ""
    except Exception as e:
        print(f"❌ WebSocket STT error: {e}")
        return ""


def listen() -> str:
    """
    Blocking wrapper around _ws_listen_once().
    Runs the async coroutine in a fresh event-loop so it can be called
    from any synchronous context, exactly like the old Whisper `listen`.
    Returns the transcribed text, or an empty string if recognition fails.
    """
    if not _websockets_available:
        return ""
    try:
        return asyncio.run(_ws_listen_once())
    except Exception as e:
        print(f"❌ listen() error: {e}")
        return ""
# ── End WebSocket STT ────────────────────────────────────────────────────────

# TTS imports
import edge_tts
import pygame
import os

os.environ["LIBCAMERA_LOG_LEVELS"] = "*:ERROR"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"


def find_usb_audio_device() -> tuple[str, str]:
    """
    Auto-detect USB audio device indices for aplay and pygame.
    Returns (aplay_device_str, card_index_str).
    Falls back to 'plughw:1,0' and '1' if detection fails.
    """
    try:
        result = subprocess.run(
            ["aplay", "-l"], capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if "usb" in line.lower() or "USB" in line:
                # Line looks like: "card 1: Device [USB Audio Device], device 0: ..."
                parts = line.split(":")
                if parts[0].startswith("card"):
                    card_idx = parts[0].strip().split()[-1]
                    print(f"🎤 USB audio card detected: card {card_idx}")
                    # sysdefault allows shared access and avoids ESTRPIPE (error 524)
                    # that plughw causes when the device is reopened quickly.
                    return f"sysdefault:CARD={card_idx}", card_idx
    except Exception as e:
        print(f"⚠️ USB audio detection error: {e}")

    print("⚠️ USB audio device not found via aplay, defaulting to plughw:1,0")
    return "plughw:1,0", "1"


def _aplay_with_retry(device: str, filepath: str, retries: int = 3) -> None:
    """Play audio via aplay, retrying on transient ALSA errors (e.g. error 524)."""
    for attempt in range(retries):
        ret = os.system(f"aplay -D {device} {filepath} 2>/dev/null")
        if ret == 0:
            return
        if attempt < retries - 1:
            time.sleep(0.3)
    # Final fallback: system default ALSA device
    os.system(f"aplay {filepath} 2>/dev/null")


class BuddyPi:
    """Pi Hardware Service - connects to brain via API"""

    def __init__(self, config: Optional[Config] = None, llm_service_url: Optional[str] = None):
        self.config = config or Config.from_env()
        self.llm_service_url = llm_service_url or self.config.llm_service_url
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Detect USB audio device once at startup
        self.aplay_device, self.usb_card_index = find_usb_audio_device()
        print(f"🔊 Using audio device: {self.aplay_device}")

        # Initialize camera
        self._init_camera()
        self.csi_process = None

        # Initialize face components
        self.detector = FaceDetector(self.config.cascade_path, self.config)
        self.face_recognizer = FaceRecognizer(self.config.model_path, self.config)
        self.stability = StabilityTracker(self.config)

        # Initialize object detection
        print(f"🔍 Initializing object detection...")
        self.object_detector = ObjectDetector(confidence_threshold=0.6)
        self.stable_objects = set()
        self.persistent_objects = set()
        print(f"✅ Object detection ready")

        # Initialize hardware controllers
        self.servo = ServoController()
        self.motors = MotorController(port=ARDUINO_PORT, baud=ARDUINO_BAUD,
                                      auto_stop_sec=MOTOR_MOVE_DURATION)
        # Notify on obstacle detection from Arduino
        self.motors.set_obstacle_callback(self._on_obstacle)

        # Face tracking state
        self.face_lost_count = 0
        self.max_face_lost = 10

        # Initialize speech (WebSocket STT + Edge TTS)
        self._init_speech()

        # Application state
        self.last_recognition_time = 0
        self.last_object_detection_time = 0
        self.last_object_clear_time = time.time()
        self.current_detections = []
        self.running = False
        self.is_speaking = False
        self.active_user = None
        self.sleep_mode = False
        self.failed_listen_attempts = 0
        self.max_failed_attempts = 4
        self.frame_count = 0

        # Face recognition robustness
        self.recognition_attempts = {}

        # Obstacle flag polled by conditional moves
        self._obstacle_detected = False
        self.max_attempts = 5
        self.recognition_threshold = 0.4

        print("🤖 Buddy Pi Hardware Ready")

        # Initial greeting
        self._startup_greeting()

    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=self.config.log_format
        )

    def _init_camera(self):
        """Initialize camera (CSI or USB) - Pi 4B optimized"""
        self.csi_frame_path = "/tmp/csi_frame.jpg"
        self.use_csi = False
        self.cap = None
        self.csi_process = None

        # Check CSI availability
        if shutil.which("rpicam-hello"):
            print("📷 Starting CSI camera helper...")
            try:
                self.csi_process = subprocess.Popen(
                    ["/usr/bin/python3", "csi_camera_helper.py"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                # Wait up to 5 seconds for first frame
                for _ in range(50):
                    if os.path.exists(self.csi_frame_path):
                        self.use_csi = True
                        print("✅ CSI camera ready")
                        atexit.register(self._cleanup_csi)
                        return
                    time.sleep(0.1)

                print("⚠️ CSI frame not received, stopping helper")
                self.csi_process.terminate()
                self.csi_process = None
            except Exception as e:
                print(f"⚠️ CSI camera failed: {e}")
                if self.csi_process:
                    self.csi_process.terminate()
                    self.csi_process = None

        # Fallback: USB webcam
        print("📷 Searching for USB camera...")
        for idx in (0, 1, 2):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.camera_buffer_size)
                cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    print(f"✅ USB camera found at /dev/video{idx}")
                    return
            cap.release()

        raise RuntimeError("❌ No camera found")

    def _read_frame(self):
        """Read frame from CSI or USB camera"""
        if self.use_csi and os.path.exists(self.csi_frame_path):
            frame = cv2.imread(self.csi_frame_path)
            return (True, frame) if frame is not None else (False, None)

        if self.cap:
            return self.cap.read()

        return False, None

    def _init_speech(self):
        """Initialize speech recognition (WebSocket STT) and TTS (Edge TTS) for USB mic"""
        if not _websockets_available:
            print("❌ WebSocket STT not available - speech disabled")
            self.speech_enabled = False
            return

        self.tts_voice = "en-IN-NeerjaNeural"
        # speech_enabled depends only on websockets being importable.
        # TTS playback uses aplay (os.system), not pygame — so pygame
        # failure must never disable speech.
        self.speech_enabled = True

        # Best-effort pygame init; non-fatal if it fails.
        try:
            os.environ["SDL_AUDIODRIVER"] = "alsa"
            os.environ["AUDIODEV"] = f"plughw:{self.usb_card_index},0"
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        except Exception as e:
            print(f"⚠️ pygame mixer init failed (non-fatal, aplay handles TTS): {e}")

        print(f"✅ Speech system ready (WebSocket STT @ {STT_SERVER_IP}:{STT_PORT} + Edge TTS via USB mic, card {self.usb_card_index})")

    def _startup_greeting(self):
        """Detect and greet person on startup with robust recognition"""
        try:
            print("Starting camera and face recognition...")
            time.sleep(1.0)

            recognition_attempts = {}

            for attempt in range(50):
                ret, frame = self._read_frame()
                if not ret:
                    continue

                faces = self.detector.detect(frame)

                if faces:
                    largest_face = self.detector.get_largest_face(faces)
                    is_stable = self.stability.update(largest_face)
                    x, y, w, h = largest_face

                    processed = self._draw_visualization(frame, faces)
                    cv2.imshow('Buddy Vision', processed)
                    cv2.waitKey(1)

                    if w > 80 and h > 80 and is_stable:
                        face_roi = frame[y:y+h, x:x+w]
                        name, confidence = self.face_recognizer.recognize(face_roi)

                        if name != "Unknown" and confidence > self.recognition_threshold:
                            if name not in recognition_attempts:
                                recognition_attempts[name] = 0
                            recognition_attempts[name] += 1

                            if recognition_attempts[name] >= 3:
                                self.active_user = name
                                print(f"✅ Recognized: {name}")

                                response = self._call_brain_service(
                                    "Hello! Nice to see you!",
                                    recognized_user=name
                                )
                                self._display_response(response)
                                return

                        total_attempts = sum(recognition_attempts.values())
                        if total_attempts >= self.max_attempts and not self.active_user:
                            print("❓ Unknown person after multiple attempts")
                            response = self._call_brain_service(
                                "I can see someone but don't recognize them. Greet them politely and ask their name.",
                                recognized_user=None
                            )
                            self._display_response(response)
                            return

                time.sleep(0.1)

            # Fallback greeting
            print("Starting general interaction")
            response = self._call_brain_service("Hello! I'm ready to chat!")
            self._display_response(response)

        except Exception as e:
            self.logger.error(f"Startup greeting error: {e}")
            print("Buddy: Hello! I'm ready to chat!")

    def _call_brain_service(self, user_input: str, recognized_user: Optional[str] = None) -> dict:
        """Call remote brain service"""
        try:
            user_lower = user_input.lower()
            identity_patterns = [
                'who am i', 'who is this', 'who is here', 'who do you see',
                'recognize me', 'recognize this', 'can you see me', 'do you know me',
                'identify me', 'identify this', 'scan face', 'scan me'
            ]
            if any(pattern in user_lower for pattern in identity_patterns):
                recognized_user = self._force_face_recognition()
                if recognized_user:
                    print(f"🔍 Forced recognition: {recognized_user}")

            objects = []
            if any(phrase in user_input.lower() for phrase in
                  ['what is', 'what do you see', 'in my hand', 'holding', 'show you']):
                ret, frame = self._read_frame()
                if ret:
                    fresh_detections = self.object_detector.detect(frame)
                    if fresh_detections:
                        objects = [det['name'] for det in fresh_detections]
                        self.persistent_objects.update(objects)
                        print(f"🔄 Fresh detection: {objects}")

            if not objects:
                objects = list(self.persistent_objects) if self.persistent_objects else list(self.stable_objects)

            request_data = {
                "user_input": user_input,
                "recognized_user": recognized_user,
                "objects_visible": objects
            }

            response = requests.post(
                f"{self.llm_service_url}/chat",
                json=request_data,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Brain service error: {response.status_code}")

        except Exception as e:
            self.logger.error(f"Brain service call failed: {e}")
            return {
                "reply": "Sorry, I'm having trouble thinking right now.",
                "intent": "conversation",
                "emotion": "apologetic",
                "raw_response": ""
            }

    def _force_face_recognition(self) -> Optional[str]:
        """Force immediate face recognition, can handle multiple faces"""
        try:
            ret, frame = self._read_frame()
            if not ret:
                return None

            try:
                faces = self.detector.detect(frame)
            except cv2.error as e:
                print(f"Face detection error: {e}")
                return None
            except Exception as e:
                print(f"Face detection error: {e}")
                return None

            if not faces:
                return None

            recognized_names = []

            for (x, y, w, h) in faces:
                if w > 80 and h > 80:
                    try:
                        face_roi = frame[y:y+h, x:x+w]
                        name, confidence = self.face_recognizer.recognize(face_roi)

                        if name != "Unknown" and confidence > self.recognition_threshold:
                            recognized_names.append(name)
                            print(f"✅ Recognized: {name} (confidence: {confidence:.2f})")
                    except Exception as e:
                        print(f"Face recognition error: {e}")
                        continue

            if recognized_names:
                self.active_user = recognized_names[0]
                if len(recognized_names) == 1:
                    return recognized_names[0]
                else:
                    names_str = ", ".join(recognized_names)
                    print(f"👥 Multiple people: {names_str}")
                    return recognized_names[0]

            return None

        except Exception as e:
            print(f"Force recognition error: {e}")
            return None

    def _greet_recognized_user(self, name: str):
        """Send a greeting to the brain for a newly recognised user (runs in thread)."""
        try:
            response = self._call_brain_service(
                f"You just recognised {name}. Greet them warmly by name.",
                recognized_user=name
            )
            self._display_response(response)
        except Exception as e:
            print(f"Greeting error: {e}")

    def _process_frame(self, frame: np.ndarray) -> tuple:
        """Process frame for face detection and recognition"""
        try:
            faces = self.detector.detect(frame)
        except (cv2.error, Exception) as e:
            print(f"Face detection error: {e}")
            faces = []

        name = None
        confidence = 0.0
        face_detected = len(faces) > 0

        if face_detected:
            largest_face = self.detector.get_largest_face(faces)
            is_stable = self.stability.update(largest_face)

            current_time = time.time()
            if (is_stable and
                    (current_time - self.last_recognition_time) > 30.0):

                x, y, w, h = largest_face
                if w > self.config.min_face_size[0] and h > self.config.min_face_size[1]:
                    face_roi = frame[y:y+h, x:x+w]
                    name, confidence = self.face_recognizer.recognize(face_roi)
                    print(f"🔍 Recognition attempt → {name} ({confidence:.2f})")

                    if name != "Unknown" and confidence > self.recognition_threshold:
                        if self.active_user != name:
                            print(f"✅ New user detected: {name}")
                            self.active_user = name
                            # Greet the newly recognised person
                            threading.Thread(
                                target=self._greet_recognized_user,
                                args=(name,),
                                daemon=True
                            ).start()
                        else:
                            print(f"✅ Still with: {name}")
                    else:
                        print(f"❓ Face not recognised (name={name}, conf={confidence:.2f})")

                    self.last_recognition_time = current_time
        else:
            self.stability.reset()

        current_time = time.time()
        if (current_time - self.last_object_detection_time) > 5.0:
            self.current_detections = self._process_objects(frame)
            self.last_object_detection_time = current_time

        if (current_time - self.last_object_clear_time) > 60.0:
            print(f"🧹 Clearing old objects: {self.persistent_objects}")
            self.persistent_objects.clear()
            self.last_object_clear_time = current_time

        processed = self._draw_visualization(frame, faces, name, confidence, self.current_detections)

        return processed, face_detected, name, confidence

    def _process_objects(self, frame):
        """Process object detection with debug output"""
        try:
            detections = self.object_detector.detect(frame)
            if detections:
                current_objects = set([det['name'] for det in detections])
                confidence_info = [(det['name'], det['confidence']) for det in detections]
                print(f"🔍 Objects detected: {confidence_info}")

                self.stable_objects = current_objects
                self.persistent_objects.update(current_objects)
                return detections
            else:
                if self.stable_objects:
                    print(f"🚫 No objects detected this frame")
                self.stable_objects = set()
                return []
        except Exception as e:
            print(f"⚠️ Object detection error: {e}")
            self.stable_objects = set()
            return []

    def speak(self, text):
        """Convert text to speech using Edge TTS"""
        if not self.speech_enabled or not text:
            return

        def _speak():
            try:
                self.is_speaking = True
                asyncio.run(self._generate_and_play_speech(text))
                self.is_speaking = False
            except Exception as e:
                print(f"TTS Error: {e}")
                self.is_speaking = False

        threading.Thread(target=_speak, daemon=True).start()

    async def _generate_and_play_speech(self, text):
        """Generate speech using Edge TTS and play via USB audio device.

        Edge TTS streams MP3 data. aplay only handles raw PCM/WAV, so feeding
        it MP3 bytes directly produces buzzing. We convert MP3 -> WAV via
        ffmpeg first, falling back to mpg123 (native MP3 player) if needed.
        """
        try:
            communicate = edge_tts.Communicate(text, self.tts_voice)

            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if not audio_data:
                return

            mp3_path = "/tmp/tts_output.mp3"
            wav_path = "/tmp/tts_output.wav"

            with open(mp3_path, "wb") as f:
                f.write(audio_data)

            # Option 1: ffmpeg converts MP3 -> WAV (signed 16-bit PCM, 22050 Hz, stereo)
            if shutil.which("ffmpeg"):
                ret = os.system(
                    f"ffmpeg -y -loglevel quiet"
                    f" -i {mp3_path}"
                    f" -ar 22050 -ac 2 -sample_fmt s16"
                    f" {wav_path}"
                )
                if ret == 0:
                    _aplay_with_retry(self.aplay_device, wav_path)
                    return

            # Option 2: mpg123 decodes and plays MP3 directly to the ALSA card
            if shutil.which("mpg123"):
                os.system(
                    f"mpg123 -q -a hw:{self.usb_card_index},0 {mp3_path} 2>/dev/null"
                )
                return

            # Last resort — will still buzz, but at least won't crash
            print("⚠️ ffmpeg and mpg123 not found — run: sudo apt install ffmpeg")
            _aplay_with_retry(self.aplay_device, mp3_path)

        except Exception as e:
            print(f"Edge TTS Error: {e}")

    def listen_for_speech(self, timeout=6.0):
        """Listen for speech using WebSocket STT via USB microphone"""
        if not self.speech_enabled or not _websockets_available:
            print("⚠️ Speech system disabled")
            return ""

        try:
            print("🎤 Listening (WebSocket STT)...")

            # listen() records audio via sounddevice, resamples, and sends
            # it to the remote WebSocket STT server, returning the transcript.
            text = listen()

            if text and len(text) > 2:
                print(f"🎤 You said: '{text}'")
                return text
            else:
                print("🔇 No speech detected")
                return ""

        except Exception as e:
            print(f"❌ Audio error: {e}")
            return ""

    def _draw_visualization(self, frame, faces, name=None, confidence=0.0, detections=None):
        """Draw UI overlays with object detection boxes"""
        if detections:
            frame = self.object_detector.draw_detections(frame, detections)

        for (x, y, w, h) in faces:
            color = (0, 255, 0) if self.stability.is_stable else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            if name and name != "Unknown":
                label = f"{name} ({confidence:.0%})"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if self.active_user:
            status = f"💬 Chatting with: {self.active_user}"
            color = (0, 255, 0)
        else:
            status = "🔍 Looking for people"
            color = (0, 255, 255)

        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if self.persistent_objects:
            objects_text = f"Objects: {', '.join(list(self.persistent_objects)[:3])}"
            cv2.putText(frame, objects_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if self.stable_objects:
            current_text = f"Current: {', '.join(list(self.stable_objects)[:2])}"
            cv2.putText(frame, current_text, (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return frame

    def _display_response(self, response: dict):
        """Display AI response and handle TTS"""
        if not response:
            return

        reply = response.get("reply", "")
        intent = response.get("intent", "conversation")
        raw_response = response.get("raw_response", "")

        print(f"\nBuddy: {raw_response if raw_response else reply}")

        if intent != "conversation":
            print(f"[INTENT: {intent}]")

            if intent in [
                "move_forward", "move_backward", "move_left", "move_right", "stop",
                "continuous_forward", "continuous_backward",
                "continuous_left", "continuous_right",
            ]:
                self._execute_movement(intent)

        self.speak(reply)

        # ── Emotion-based physical expression ─────────────────────────────────
        emotion = response.get("emotion", "")
        if emotion and self.motors.is_connected():
            # Delay slightly so movement starts as speech begins
            threading.Timer(0.3, lambda: self.motors.emotion_move(emotion)).start()

        words = len(reply.split())
        chars = len(reply)
        reading_time = max(2.0, (words * 0.3) + (chars * 0.05))
        reading_time = min(reading_time, 8.0)

        threading.Timer(reading_time, self._delayed_listening).start()

    def _execute_movement(self, intent: str):
        """Execute timed, continuous, or conditional movement."""
        if not self.motors.is_connected():
            print(f"⚠️ Motors not connected — ignoring: {intent}")
            return

        print(f"🤖 Executing: {intent}")

        timed_map = {
            "move_forward":  lambda: self.motors.move_forward(MOTOR_MOVE_DURATION),
            "move_backward": lambda: self.motors.move_backward(MOTOR_MOVE_DURATION),
            "move_left":     lambda: self.motors.turn_left(MOTOR_MOVE_DURATION),
            "move_right":    lambda: self.motors.turn_right(MOTOR_MOVE_DURATION),
        }
        continuous_map = {
            "continuous_forward":  lambda: self.motors.move_continuous("F"),
            "continuous_backward": lambda: self.motors.move_continuous("B"),
            "continuous_left":     lambda: self.motors.move_continuous("L"),
            "continuous_right":    lambda: self.motors.move_continuous("R"),
        }
        # Conditional: move until obstacle_detected flag is set by Arduino
        conditional_map = {
            "conditional_forward":  lambda: self._run_conditional("F"),
            "conditional_backward": lambda: self._run_conditional("B"),
            "conditional_left":     lambda: self._run_conditional("L"),
            "conditional_right":    lambda: self._run_conditional("R"),
        }

        if intent == "stop":
            self.motors.stop()
        elif intent in conditional_map:
            conditional_map[intent]()
        elif intent in continuous_map:
            continuous_map[intent]()
        elif intent in timed_map:
            timed_map[intent]()
        else:
            print(f"⚠️ Unknown intent: {intent}")

    def _run_conditional(self, cmd: str):
        """Start a conditional move — let Arduino's obstacle detection stop it."""
        self._obstacle_detected = False   # reset flag before starting
        def obstacle_condition():
            return self._obstacle_detected
        self.motors.move_until(cmd, obstacle_condition)

    def _play_thinking_sound(self):
        """Play thinking sound while processing"""
        thinking_sounds = ["Hmm", "Let me think", "Umm", "Well"]
        import random
        sound = random.choice(thinking_sounds)
        self.speak(sound)
        time.sleep(0.3)

    def _delayed_listening(self):
        """Start listening after a delay — waits for TTS to finish first."""
        if not self.running or self.sleep_mode:
            return

        # Wait until TTS finishes so the mic doesn't pick up speaker output.
        wait_start = time.time()
        while self.is_speaking and (time.time() - wait_start) < 15.0:
            time.sleep(0.1)
        # Extra buffer so ALSA fully releases the device after aplay exits
        time.sleep(0.4)

        user_text = self.listen_for_speech(timeout=6.0)

        if user_text:
            self.failed_listen_attempts = 0
            self._play_thinking_sound()
            threading.Thread(target=self._process_input, args=(user_text,), daemon=True).start()
        else:
            self.failed_listen_attempts += 1
            print(f"🔇 No input attempt {self.failed_listen_attempts}/{self.max_failed_attempts}")

            if self.failed_listen_attempts >= self.max_failed_attempts:
                self._enter_sleep_mode()
            elif self.running:
                threading.Timer(1.0, self._delayed_listening).start()

    # Maps of voice keywords → intent strings for local motor detection.
    # These fire immediately without waiting for brain intent classification,
    # ── Direction extraction ─────────────────────────────────────────────────
    # Maps text fragments → Arduino command char
    _DIR_MAP = {
        "forward": "F", "ahead": "F", "front": "F",
        "backward": "B", "back": "B", "reverse": "B", "behind": "B",
        "left": "L",
        "right": "R",
    }

    # ── Condition keywords that mean "stop when X happens" ───────────────────
    _CONDITION_KEYWORDS = [
        "until obstacle", "until you hit", "until you see", "until you detect",
        "until wall", "until something", "until blocked",
        "till obstacle", "till you hit", "till you see", "till you detect",
        "until person", "until someone", "until face",
        "keep going", "keep moving", "keep driving", "continuously",
        "non-stop", "without stopping", "indefinitely",
        "until i say stop", "until i tell you to stop",
    ]

    # ── Timed single-burst phrases ────────────────────────────────────────────
    _MOVE_KEYWORDS = {
        "move forward": "move_forward", "go forward": "move_forward",
        "move ahead":   "move_forward", "go ahead":   "move_forward",
        "move backward":"move_backward","go backward":"move_backward",
        "move back":    "move_backward","go back":    "move_backward",
        "reverse":      "move_backward",
        "move left":    "move_left",    "go left":    "move_left",
        "turn left":    "move_left",
        "move right":   "move_right",   "go right":   "move_right",
        "turn right":   "move_right",
        "stop":  "stop", "halt": "stop", "freeze": "stop",
    }

    # ── Confirmations (no brain, instant) ────────────────────────────────────
    _MOTOR_CONFIRMATIONS = {
        "conditional_forward":  "Moving forward until the condition is met.",
        "conditional_backward": "Moving backward until the condition is met.",
        "conditional_left":     "Turning left until the condition is met.",
        "conditional_right":    "Turning right until the condition is met.",
        "continuous_forward":   "Moving forward.",
        "continuous_backward":  "Moving backward.",
        "continuous_left":      "Turning left.",
        "continuous_right":     "Turning right.",
        "move_forward":         "Moving forward.",
        "move_backward":        "Moving backward.",
        "move_left":            "Turning left.",
        "move_right":           "Turning right.",
        "stop":                 "Stopped.",
    }

    def _parse_movement_intent(self, text: str):
        """
        Parse free-form movement text into an intent string.
        Returns intent string or None if no movement detected.

        Examples:
          "move forward until you detect an obstacle" → "conditional_forward"
          "keep going left"                           → "continuous_left"
          "move forward"                              → "move_forward"
          "stop"                                      → "stop"
        """
        t = text.lower()

        # Stop check first
        if any(w in t for w in ("stop", "halt", "freeze")):
            return "stop"

        # Extract direction
        direction = None
        for word, cmd in self._DIR_MAP.items():
            if word in t:
                direction = cmd
                break

        if direction is None:
            return None

        dir_name = {
            "F": "forward", "B": "backward", "L": "left", "R": "right"
        }[direction]

        # Check for condition/continuous keyword
        if any(kw in t for kw in self._CONDITION_KEYWORDS):
            return f"conditional_{dir_name}"

        # Plain timed move — match existing keyword dict
        for phrase, intent in self._MOVE_KEYWORDS.items():
            if phrase in t:
                return intent

        return None

    def _process_input(self, user_text):
        """Process user input.

        1. Try to parse a movement intent locally (instant, no brain call).
        2. If it's a movement command → execute + short TTS + return.
        3. Otherwise → brain service for conversation / questions.
        """
        try:
            text_lower = user_text.lower()

            # ── Step 1: parse movement intent from free-form text ─────────────
            matched_intent = self._parse_movement_intent(text_lower)

            # ── Step 2: motor command — execute locally, skip brain entirely ──
            if matched_intent is not None:
                print(f"🎙️ Motor intent: '{matched_intent}'")
                self._execute_movement(matched_intent)
                confirmation = self._MOTOR_CONFIRMATIONS.get(matched_intent, "On it.")
                self.speak(confirmation)
                threading.Timer(1.5, self._delayed_listening).start()
                return   # ← do NOT call brain

            # ── Step 3: non-motor — register name if introduced ───────────────
            if any(phrase in text_lower for phrase in ['my name is', 'i am', "i'm"]):
                name = self._extract_name(user_text)
                if name and len(name.split()) == 1 and name.isalpha() and len(name) > 1:
                    if self.active_user != name:
                        if hasattr(self, 'last_frame') and self.last_frame is not None:
                            faces = self.detector.detect(self.last_frame)
                            if faces:
                                largest_face = self.detector.get_largest_face(faces)
                                x, y, w, h = largest_face
                                if w > 80 and h > 80:
                                    face_roi = self.last_frame[y:y+h, x:x+w]
                                    if self.face_recognizer.add_face(name, face_roi):
                                        self.active_user = name
                                        print(f"Registered new face for: {name}")

            # ── Step 5: everything else goes to the brain ─────────────────────
            response = self._call_brain_service(user_text, self.active_user)
            self._display_response(response)

        except Exception as e:
            print(f"Error processing input: {e}")

    def _on_obstacle(self):
        """Called when Arduino reports an obstacle."""
        print("⚠️ Obstacle detected — all moves cancelled")
        self._obstacle_detected = True   # signals conditional move loops
        if not self.is_speaking:
            self.speak("Obstacle detected. I've stopped.")

    def _enter_sleep_mode(self):
        """Trigger sleep mode - ends conversation loop"""
        print("\n😴 Entering sleep mode...")
        self.sleep_mode = True

    def _wake_up_and_restart(self):
        """Wake up and restart conversation loop"""
        print("😊 Waking up!")
        self.sleep_mode = False
        self.failed_listen_attempts = 0

        # Re-initialize speech in case it was disrupted during sleep loop
        self._init_speech()

        self.speak("I'm awake! Let me see who's here.")
        time.sleep(1)

        recognized_user = self._force_face_recognition()
        if recognized_user:
            self.active_user = recognized_user
            response = self._call_brain_service(
                "I just woke up. Greet them warmly.",
                recognized_user=recognized_user
            )
        else:
            response = self._call_brain_service("I just woke up. Say hello.")

        self._display_response(response)

    def _extract_name(self, text: str) -> str:
        """Extract name from introduction text"""
        text = text.lower()
        if 'my name is' in text:
            return text.split('my name is')[1].strip().title()
        elif 'i am' in text:
            return text.split('i am')[1].strip().title()
        elif "i'm" in text:
            return text.split("i'm")[1].strip().title()
        return ""

    def _cleanup_csi(self):
        """Cleanup CSI camera process"""
        if hasattr(self, 'csi_process') and self.csi_process:
            self.csi_process.terminate()

    def run(self):
        """Main application loop with independent sleep/wake cycles"""
        self.running = True

        try:
            self._conversation_loop()
        except KeyboardInterrupt:
            print("\n⚠️ Keyboard interrupt received")
            self.running = False
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
        finally:
            self.cleanup()

    def _conversation_loop(self):
        """Active conversation loop with camera and face detection"""
        print("🎯 Starting conversation loop")
        self.last_frame = None
        self.failed_listen_attempts = 0

        while self.running and not self.sleep_mode:
            ret, frame = self._read_frame()
            if not ret:
                continue

            self.last_frame = frame.copy()
            self.frame_count += 1

            if self.frame_count % 5 == 0:
                processed, face_detected, name, confidence = self._process_frame(frame)
            else:
                processed = frame

            cv2.imshow('Buddy Vision', processed)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break

            time.sleep(0.01)

        if self.sleep_mode and self.running:
            self._sleep_loop()

    def _sleep_loop(self):
        """Independent sleep loop - only wake word detection"""
        print("😴 Starting sleep loop")

        try:
            cv2.destroyAllWindows()
            print("OpenCV window closed")
        except:
            pass

        self.speak("Going to sleep. Say 'Hey Buddy' to wake me up.")
        time.sleep(2)

        print("😴 Sleep mode active. Listening for 'Hey Buddy'...")

        while self.running and self.sleep_mode:
            try:
                print("[SLEEP] Listening for wake word...")

                text = listen() if _websockets_available else ""

                if text:
                    text_lower = text.lower()
                    print(f"[SLEEP] Heard: '{text}'")

                    if any(phrase in text_lower for phrase in ['hey buddy', 'wake up', 'buddy wake up']):
                        print(f"✅ [SLEEP] Wake word detected: '{text}'")
                        self._wake_up_and_restart()
                        break
                    else:
                        print(f"❌ [SLEEP] Not a wake word: '{text}'")
                else:
                    print("🔇 [SLEEP] Could not understand audio")
                    time.sleep(0.5)

            except KeyboardInterrupt:
                print("\n⚠️ [SLEEP] Interrupted during sleep mode")
                self.running = False
                break
            except Exception as e:
                print(f"❌ [SLEEP] Wake detection error: {e}")
                time.sleep(0.5)

        if not self.sleep_mode and self.running:
            self._conversation_loop()

    def cleanup(self):
        """Clean up resources"""
        self.running = False

        # Stop motors and close Serial before anything else
        if hasattr(self, 'motors'):
            try:
                self.motors.stop()
                self.motors.cleanup()
            except Exception:
                pass

        try:
            requests.post(f"{self.llm_service_url}/clear", timeout=5)
        except:
            pass

        if self.use_csi and hasattr(self, 'csi_process') and self.csi_process:
            self.csi_process.terminate()

        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

        try:
            cv2.destroyAllWindows()
            pygame.mixer.quit()
        except:
            pass

        print("\nBuddy Pi: Goodbye!")


def main():
    """Entry point"""
    try:
        config = Config.from_env()

        llm_url = config.llm_service_url

        print(f"🤖 Starting Buddy Pi...")
        print(f"🔗 LLM Service: {llm_url}")
        print(f"📷 Camera: {config.camera_index}")

        buddy = BuddyPi(config)

        def shutdown(sig, frame):
            print("\n⚠️ Shutting down...")
            buddy.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        buddy.run()

        return 0

    except Exception as e:
        logging.error(f"Startup failed: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
