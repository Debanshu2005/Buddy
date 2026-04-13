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

# Hardware modules only
from core.config import Config
from core.states import BuddyState, StateManager
from vision.face_detector import FaceDetector
from vision.face_recognizer import FaceRecognizer
from core.stability_tracker import StabilityTracker
from vision.objrecog.obj import ObjectDetector
from hardware.servo_controller import ServoController
from hardware.motor_controller import MotorController

# ── WebSocket STT ────────────────────────────────────────────────────────────
# Recording uses arecord (ALSA command-line tool) instead of sounddevice so
# that PortAudio is never touched by the STT path.  This prevents the
# PaUnixMutex_Terminate assertion crash that occurred when sounddevice was
# initialised/destroyed across multiple asyncio event loops simultaneously.
import asyncio
import tempfile
import struct
from scipy.signal import resample_poly

STT_SERVER_IP  = "192.168.0.169"
STT_PORT       = 8765
MIC_RATE       = 48000   # Native mic sample rate (arecord)
TARGET_RATE    = 16000   # Rate the STT server expects
CHUNK_DURATION = 5       # Seconds per recording — longer = more time for far speech
MIC_GAIN       = 4.0     # Software amplification (1.0 = none, 4.0 = 4× louder)
                         # Increase if mic still can't hear from far (try 6.0 or 8.0)
VAD_THRESHOLD  = 0.002   # RMS energy below this = silence, skip STT call
                         # Lower if it skips real speech; raise if it sends silence

try:
    import websockets as _websockets_lib
    _websockets_available = True
except ImportError as e:
    print(f"❌ Failed to import websockets: {e}")
    _websockets_available = False


def _record_audio_arecord(device: str = None) -> np.ndarray:
    """
    Record CHUNK_DURATION seconds via arecord using the detected USB mic.
    Falls back through a chain of devices if the primary fails.
    Returns float32 numpy array normalised to [-1,1], or empty on failure.
    """
    import wave

    primary = device or _mic_device
    card    = _mic_card_index

    # Build fallback chain from the detected device
    devices = [
        primary,                      # e.g. plughw:2,0
        f"hw:{card},0",               # raw hw
        f"plughw:{card},0",           # plug layer
        "default",                    # system default
    ]
    # Deduplicate while preserving order
    seen = set()
    devices = [d for d in devices if not (d in seen or seen.add(d))]

    wav_file = tempfile.mktemp(suffix=".wav")

    for dev in devices:
        cmd = [
            "arecord",
            "-D", dev,
            "-f", "S16_LE",
            "-r", str(MIC_RATE),
            "-c", "1",
            "-d", str(CHUNK_DURATION),
            wav_file
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=CHUNK_DURATION + 3
            )
            if result.returncode == 0 and os.path.exists(wav_file):
                break   # success
            err = (result.stderr or "").strip()
            if err:
                print(f"⚠️ arecord [{dev}]: {err.splitlines()[-1]}")
            time.sleep(0.2)
        except subprocess.TimeoutExpired:
            print(f"⚠️ arecord timed out on {dev}")
        except Exception as e:
            print(f"⚠️ arecord error [{dev}]: {e}")
    else:
        print("❌ arecord failed on all devices — run: arecord -l  to check mic")
        return np.array([], dtype=np.float32)

    try:
        with wave.open(wav_file, "rb") as wf:
            raw     = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            samples /= 32768.0   # normalise to [-1, 1]

        os.remove(wav_file)

        # ── Software gain boost ───────────────────────────────────────────────
        # Amplifies quiet far-field audio before sending to STT.
        # np.clip prevents clipping distortion above ±1.0.
        samples = np.clip(samples * MIC_GAIN, -1.0, 1.0)

        # ── Voice activity detection (VAD) ───────────────────────────────────
        # Skip STT entirely if the recording is just silence — saves time and
        # prevents the STT server returning garbage transcriptions.
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < VAD_THRESHOLD:
            return np.array([], dtype=np.float32)   # silent → skip

        return samples
    except Exception as e:
        print(f"❌ WAV read error: {e}")
        return np.array([], dtype=np.float32)


# Set at runtime by BuddyPi.__init__ after USB detection
# mic=card3 (USB PnP Sound Device), speaker=card0 (bcm2835 3.5mm jack)
_mic_card_index: str = "3"
_mic_device:     str = "plughw:3,0"   # USB mic (arecord)
_spk_device:     str = "plughw:0,0"   # bcm2835 3.5mm speaker (aplay)

async def _ws_listen_once() -> str:
    """
    Record one audio chunk via arecord, resample, send to WebSocket STT server,
    return transcript.  PortAudio / sounddevice not used anywhere here.
    """
    uri = f"ws://{STT_SERVER_IP}:{STT_PORT}"
    try:
        # Recording is blocking — run in executor so it doesn't block the loop
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None, _record_audio_arecord
        )

        if audio.size == 0:
            return ""

        # Resample 48 kHz → 16 kHz
        audio_16k = resample_poly(audio, TARGET_RATE, MIC_RATE).astype(np.float32)

        async with _websockets_lib.connect(uri) as websocket:
            await websocket.send(audio_16k.tobytes())
            text = await websocket.recv()
            return text.strip() if text else ""
    except Exception as e:
        print(f"❌ WebSocket STT error: {e}")
        return ""


def listen() -> str:
    """
    Blocking wrapper — safe to call from any thread.
    Each call gets its own fresh event loop so there is no cross-thread
    event-loop conflict with the TTS asyncio path.
    """
    if not _websockets_available:
        return ""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_ws_listen_once())
        finally:
            loop.close()
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




def _scan_alsa_cards(tool: str) -> list[tuple[str, str, str]]:
    """Scan ALL ALSA cards (not just USB)."""
    results = []
    try:
        r = subprocess.run([tool, "-l"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.startswith("card"):
                try:
                    card_num = line.split(":")[0].strip().split()[-1]
                    short = line.split(":")[1].strip().split("[")[0].strip()
                    card_name = short.replace(" ", "_")
                    dev_num = line.split("device")[-1].strip().split(":")[0].strip()
                    results.append((card_num, card_name, dev_num))
                except Exception:
                    pass
    except Exception as e:
        print(f"⚠️ {tool} -l error: {e}")
    return results


def _is_bluetooth_connected() -> bool:
    """Check if Bluetooth audio is connected."""
    try:
        result = subprocess.run(
            ["bluetoothctl", "devices", "Connected"],
            capture_output=True, text=True, timeout=2
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def find_usb_audio_device() -> tuple[str, str]:
    """
    Find playback device with priority:
      1. bcm2835 Headphones (3.5mm jack) if physically connected
      2. Bluetooth audio
      3. USB audio
      4. HDMI
      5. bcm2835 fallback
    """
    cards = _scan_alsa_cards("aplay")

    # Priority 1: 3.5mm jack (only if something plugged in)
    for card_num, card_name, dev_num in cards:
        name_low = card_name.lower()
        if "bcm2835" in name_low or "headphone" in name_low:
            try:
                result = subprocess.run(
                    ["amixer", "-c", card_num, "get", "Headphone"],
                    capture_output=True, text=True, timeout=2
                )
                if "Front Left:" in result.stdout or "Front Right:" in result.stdout:
                    print(f"🔊 3.5mm jack: card {card_num} ({card_name}), device {dev_num}")
                    return f"plughw:{card_num},{dev_num}", card_num
                else:
                    print(f"⚠️ 3.5mm exists but nothing plugged — checking Bluetooth...")
            except Exception:
                print(f"⚠️ 3.5mm detection failed — checking Bluetooth...")

    # Priority 2: Bluetooth
    if _is_bluetooth_connected():
        for card_num, card_name, dev_num in cards:
            name_low = card_name.lower()
            if "blue" in name_low or "bt" in name_low or "a2dp" in name_low:
                print(f"🔊 Bluetooth: card {card_num} ({card_name}), device {dev_num}")
                return f"plughw:{card_num},{dev_num}", card_num

    # Priority 3: USB
    for card_num, card_name, dev_num in cards:
        name_low = card_name.lower()
        if "usb" in name_low or "uac" in name_low or "pnp" in name_low:
            print(f"🔊 USB: card {card_num} ({card_name}), device {dev_num}")
            return f"plughw:{card_num},{dev_num}", card_num

    # Priority 4: HDMI
    for card_num, card_name, dev_num in cards:
        name_low = card_name.lower()
        if "hdmi" in name_low or "vc4" in name_low:
            print(f"🔊 HDMI: card {card_num} ({card_name}), device {dev_num}")
            return f"plughw:{card_num},{dev_num}", card_num

    # Fallback: bcm2835 anyway
    for card_num, card_name, dev_num in cards:
        name_low = card_name.lower()
        if "bcm2835" in name_low:
            print(f"🔊 Fallback bcm2835: card {card_num}")
            return f"plughw:{card_num},{dev_num}", card_num

    print("⚠️ No audio device — defaulting to plughw:0,0")
    return "plughw:0,0", "0"



def find_usb_mic_device() -> tuple[str, str, str]:
    """
    Auto-detect USB microphone for recording (arecord).
    Returns (arecord_device_str, card_number, device_number).

    Tries arecord -l first. If nothing found, falls back to the same
    card as the playback device (common for USB audio dongles).
    """
    cards = _scan_alsa_cards("arecord")
    if cards:
        card_num, card_name, dev_num = cards[0]
        print(f"🎤 USB mic card detected: card {card_num} ({card_name}), device {dev_num}")
        return f"plughw:{card_num},{dev_num}", card_num, dev_num

    print("⚠️ No USB capture device found in arecord -l — using card 3,0 as fallback")
    return "plughw:3,0", "3", "0"


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
        print(f"🔊 Playback device : {self.aplay_device}")

        # Detect USB mic separately (may be different card on some hardware)
        self.arecord_device, self.mic_card_index, self.mic_dev_index =             find_usb_mic_device()
        print(f"🎤 Capture device  : {self.arecord_device}")

        # Share with module-level STT recording function
        global _mic_card_index, _mic_device
        _mic_card_index = self.mic_card_index
        _mic_device     = self.arecord_device

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

        # Initialize hardware controllers - servo first
        try:
            self.servo = ServoController()
            self.servo_enabled = True
            self._servo_looking_up = False  # Track servo state for face detection
        except Exception as e:
            print(f"⚠️ Servo init failed: {e}")
            self.servo = None
            self.servo_enabled = False

        self.motors = MotorController(port=ARDUINO_PORT, baud=ARDUINO_BAUD)
        # Notify on obstacle detection from Arduino
        self.motors.set_obstacle_callback(self._on_obstacle)
        self.motors.set_clear_callback(self._on_clear)

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
        self._awaiting_name = False  # True when buddy asked a stranger for their name

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
                    ["/usr/bin/python3", str(Path(__file__).resolve().parent / "hardware" / "csi_camera_helper.py")],
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
        """On startup: recognize known face and greet by name, or ask stranger for their name"""
        try:
            print("Starting camera and face recognition...")
            time.sleep(1.0)

            recognition_counts = {}

            for _ in range(60):
                ret, frame = self._read_frame()
                if not ret:
                    continue

                faces = self.detector.detect(frame)
                if faces:
                    largest_face = self.detector.get_largest_face(faces)
                    is_stable = self.stability.update(largest_face)
                    x, y, w, h = largest_face

                    cv2.imshow('Buddy Vision', self._draw_visualization(frame, faces))
                    cv2.waitKey(1)

                    if w > 80 and h > 80 and is_stable:
                        face_roi = frame[y:y+h, x:x+w]
                        name, confidence = self.face_recognizer.recognize(face_roi)

                        if name != "Unknown" and confidence > self.recognition_threshold:
                            recognition_counts[name] = recognition_counts.get(name, 0) + 1
                            # Need 3 consistent recognitions to confirm
                            if recognition_counts[name] >= 3:
                                self.active_user = name
                                print(f"✅ Recognized: {name} ({confidence:.2f})")
                                response = self._call_brain_service(
                                    f"You just saw {name} and recognized them. Greet them warmly by name.",
                                    recognized_user=name
                                )
                                self._display_response(response)
                                return

                time.sleep(0.1)

            # No known face found — greet as stranger and ask for name
            print("❓ Unknown person — asking for name")
            self._awaiting_name = True
            response = self._call_brain_service(
                "You see someone but don't recognize them. Say hi and ask for their name politely.",
                recognized_user=None
            )
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

		# ── Servo: tilt camera based on face detection ────────────────────────
        if hasattr(self, 'servo_enabled') and self.servo_enabled and self.servo:
            if face_detected and self._servo_looking_up:
                # Face found → look center
                print(f"📷 Face detected → servo CENTERING (was UP)")
                self.servo.look_center(smooth=True)
                self._servo_looking_up = False
            elif not face_detected and not self._servo_looking_up:
                # No face → look up to search
                print(f"📷 No face → servo LOOKING UP (was centered)")
                self.servo.look_up(smooth=True)
                self._servo_looking_up = True
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
        """Convert text to speech using Edge TTS.
        Uses a fresh event loop per call — avoids conflicts with the STT
        event loop running in a different thread simultaneously.
        """
        if not self.speech_enabled or not text:
            return

        def _speak():
            try:
                self.is_speaking = True
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._generate_and_play_speech(text))
                finally:
                    loop.close()
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

            # listen() records via arecord, resamples, sends to WebSocket STT server.
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

            # Brain-returned movement intents
            _brain_move = {
                "move_forward": "F", "move_backward": "B",
                "move_left": "L",    "move_right": "R", "stop": "S",
            }
            if intent in _brain_move:
                self._execute_movement(_brain_move[intent])

        self.speak(reply)

        # ── Emotion-based physical expression ─────────────────────────────────
        emotion = response.get("emotion", "")
        print(f"🎭 Brain emotion: '{emotion}'")
        if emotion and self.motors.is_connected():
            # Small delay so physical movement starts as speech begins
            threading.Timer(0.3, lambda e=emotion: self.motors.emotion_move(e)).start()
        elif emotion and not self.motors.is_connected():
            print("⚠️ Motors not connected — skipping emotion move")

        words = len(reply.split())
        chars = len(reply)
        reading_time = max(2.0, (words * 0.3) + (chars * 0.05))
        reading_time = min(reading_time, 8.0)

        threading.Timer(reading_time, self._delayed_listening).start()

    def _execute_movement(self, cmd: str, duration: float = None):
        """
        Send a move command to the Arduino.
          cmd      : 'F','B','L','R' to move; 'S' to stop
          duration : seconds to run then auto-stop (None = run until stop/obstacle)
        """
        if not self.motors.is_connected():
            print(f"⚠️ Motors not connected — ignoring cmd: {cmd}")
            return
        if cmd == "S":
            self.motors.stop()
        else:
            self.motors.move(cmd, duration)

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
    # Direction word → Arduino command char
    _DIR_MAP = {
        "forward":  "F", "ahead": "F", "front": "F",
        "backward": "B", "back":  "B", "reverse": "B",
        "left":     "L",
        "right":    "R",
    }

    # Short confirmation spoken after local motor command (no brain call)
    _MOTOR_CONFIRMATIONS = {
        "F": "Moving forward.",
        "B": "Moving backward.",
        "L": "Turning left.",
        "R": "Turning right.",
        "S": "Stopped.",
    }

    def _parse_movement_intent(self, text: str):
        """
        Parse free-form text → (cmd, duration) or None.

        Returns:
          ("S", None)   → stop immediately
          ("F", None)   → move forward indefinitely (until stop/obstacle)
          ("F", 3.0)    → move forward for exactly 3 seconds

        Duration extracted from phrases like:
          "for 3 seconds", "for 2 minutes", "for 30 sec"
        Every other movement runs until explicitly stopped.
        """
        import re
        t = text.lower()

        # Stop
        if any(w in t for w in ("stop", "halt", "freeze", "don't move", "do not move")):
            return ("S", None)

        # Direction
        cmd = None
        for word, c in self._DIR_MAP.items():
            if word in t:
                cmd = c
                break
        if cmd is None:
            return None

        # Optional explicit duration
        duration = None
        m = re.search(r"for\s+(\d+(?:\.\d+)?)\s*(second|sec|minute|min)", t)
        if m:
            val  = float(m.group(1))
            unit = m.group(2)
            duration = val * 60 if unit.startswith("min") else val

        return (cmd, duration)

    def _process_input(self, user_text):
        """Process user input.

        1. Movement command → execute locally, skip brain.
        2. Name introduction while awaiting name → register face + greet by name.
        3. Everything else → brain service.
        """
        try:
            text_lower = user_text.lower()

            # ── Step 1: movement command ──────────────────────────────────────
            parsed = self._parse_movement_intent(text_lower)
            if parsed is not None:
                cmd, duration = parsed
                print(f"🎙️ Motor: {cmd!r}  duration={duration}")
                self._execute_movement(cmd, duration)
                self.speak(self._MOTOR_CONFIRMATIONS.get(cmd, "On it."))
                threading.Timer(1.5, self._delayed_listening).start()
                return

            # ── Step 2: name introduction → register face ─────────────────────
            if any(phrase in text_lower for phrase in ['my name is', 'i am', "i'm"]):
                name = self._extract_name(user_text)
                if name and len(name.split()) == 1 and name.isalpha() and len(name) > 1:
                    if self.active_user != name:
                        registered = False
                        if hasattr(self, 'last_frame') and self.last_frame is not None:
                            faces = self.detector.detect(self.last_frame)
                            if faces:
                                x, y, w, h = self.detector.get_largest_face(faces)
                                if w > 80 and h > 80:
                                    face_roi = self.last_frame[y:y+h, x:x+w]
                                    registered = self.face_recognizer.add_face(name, face_roi)

                        self.active_user = name
                        self._awaiting_name = False
                        print(f"✅ Registered face for: {name}")

                        response = self._call_brain_service(
                            f"The person just told you their name is {name}. {'You saved their face so you will recognize them next time. ' if registered else ''}Greet them warmly by name and start a friendly conversation.",
                            recognized_user=name
                        )
                        self._display_response(response)
                        return

            # ── Step 3: everything else → brain ───────────────────────────────
            response = self._call_brain_service(user_text, self.active_user)
            self._display_response(response)

        except Exception as e:
            print(f"Error processing input: {e}")

    def _on_obstacle(self):
        """Called when Arduino reports an obstacle."""
        print("🚧 Obstacle detected — all moves cancelled")
        self._obstacle_detected = True
        if not self.is_speaking:
            self.speak("Obstacle detected. I've stopped.")

    def _on_clear(self):
        """Called when Arduino reports the path is clear again."""
        print("✅ Path clear")
        self._obstacle_detected = False
        if not self.is_speaking:
            self.speak("Path is clear.")

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

                    if any(phrase in text_lower for phrase in
                           ['hey buddy', 'wake up', 'buddy wake up',
                            'hey buddy wake up', 'buddy']):
                        print(f"✅ [SLEEP] Wake word detected: '{text}'")
                        self._wake_up_and_restart()
                        break
                    else:
                        print(f"❌ [SLEEP] Not a wake word: '{text}'")
                        time.sleep(0.2)
                else:
                    # arecord returned empty — device busy or mic issue
                    # Wait before retrying so we don't flood the mic device
                    print("🔇 [SLEEP] No audio captured — retrying in 1s")
                    time.sleep(1.0)

            except KeyboardInterrupt:
                print("\n⚠️ [SLEEP] Interrupted during sleep mode")
                self.running = False
                break
            except Exception as e:
                print(f"❌ [SLEEP] Wake detection error: {e}")
                time.sleep(1.0)

        if not self.sleep_mode and self.running:
            self._conversation_loop()

    def cleanup(self):
        """Clean up resources"""
        self.running = False

        # Servo cleanup — center then release GPIO
        if hasattr(self, 'servo_enabled') and self.servo_enabled and self.servo:
            try:
                self.servo.look_center(smooth=False)
                time.sleep(0.3)
                self.servo.cleanup()
            except Exception:
                pass

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
