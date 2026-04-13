"""
Buddy Pi - Hardware Only Service
Face recognition, voice I/O, camera, objects - NO BRAIN
Optimized for Raspberry Pi 4B with Python 3.13
Uses WebSocket STT + streaming VAD for speech recognition
Alexa-style wake word ("Hey Buddy") with beep confirmation
"""
import os
import re
import random
import subprocess
import shutil
import atexit
import asyncio
import tempfile
import threading
import signal
import sys
import time
import logging
import requests
from typing import Optional

import cv2
import numpy as np
from scipy.signal import resample_poly

# ── Serial port for Arduino motor controller ─────────────────────────────────
ARDUINO_PORT = "/dev/ttyUSB0"   # change to /dev/ttyACM0 if needed
ARDUINO_BAUD = 115200

# ── Phone notification integration ───────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "phone_link"))
from phone_link import process_notification

# ── Hardware modules ──────────────────────────────────────────────────────────
from config import Config
from states import BuddyState, StateManager
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from stability_tracker import StabilityTracker
from objrecog.obj import ObjectDetector
# from servo_controller import ServoController
from motor_controller import MotorController

# ── TTS ───────────────────────────────────────────────────────────────────────
import edge_tts
import pygame

os.environ["LIBCAMERA_LOG_LEVELS"]        = "*:ERROR"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_DEBUG"]        = "0"

# ── Vosk local wake word detection ──────────────────────────────────────────
VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "vosk-model-small-en-us-0.15")
_vosk_model     = None
_vosk_available = False

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    _vosk_model     = VoskModel(VOSK_MODEL_PATH)
    _vosk_available = True
    print("✅ Vosk wake word model loaded")
except Exception as e:
    print(f"⚠️ Vosk not available: {e} — falling back to STT wake word")


def _vosk_listen_for_wake_word(mic_device: str, wake_words: list) -> bool:
    """
    Blocking. Reads raw 16kHz mic chunks, runs Vosk locally.
    Returns True the moment any wake word is spotted. Zero network calls.
    """
    import json
    rec  = KaldiRecognizer(_vosk_model, 16000)
    proc = subprocess.Popen(
        ["arecord", "-D", mic_device, "-f", "S16_LE", "-r", "16000", "-c", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    try:
        while True:
            raw = proc.stdout.read(8000)  # 250ms chunks
            if not raw:
                break
            if rec.AcceptWaveform(raw):
                text = json.loads(rec.Result()).get("text", "").lower()
            else:
                text = json.loads(rec.PartialResult()).get("partial", "").lower()
            if text and any(w in text for w in wake_words):
                print(f"🎯 Wake word: '{text}'")
                return True
    finally:
        proc.kill()
        proc.wait()
    return False



STT_SERVER_IP = "192.168.0.105"
STT_PORT      = 8765
MIC_RATE      = 48000   # Native mic sample rate (arecord)
TARGET_RATE   = 16000   # Rate the STT server expects

# ── Streaming VAD config ──────────────────────────────────────────────────────
MIC_GAIN           = 4.0   # Software amplification (raise to 6–8 for far-field)
VAD_SPEECH_THRESH  = 0.008 # RMS above this = speech. Raise if noise triggers it.
VAD_SILENCE_THRESH = 0.003 # RMS below this = silence. Lower if speech gets cut early.
VAD_MAX_DURATION   = 8.0   # Hard cap — stops recording after this many seconds
VAD_MIN_SPEECH     = 0.4   # Minimum seconds of detected speech before accepting
VAD_SILENCE_AFTER  = 0.3   # Seconds of silence after speech before cutting off
_VAD_CHUNK_SECS    = 0.1   # Size of each recorded chunk (100 ms)

# ── Wake word / conversation config ──────────────────────────────────────────

try:
    import websockets as _websockets_lib
    _websockets_available = True
except ImportError as e:
    print(f"❌ Failed to import websockets: {e}")
    _websockets_available = False

# ── Module-level mic globals (set at runtime by BuddyPi.__init__) ─────────────
_mic_card_index:      str        = "3"
_mic_device:          str        = "plughw:3,0"
_working_arecord_dev: str | None = None   # cached after first successful probe


# ─────────────────────────────────────────────────────────────────────────────
#  STREAMING VAD RECORDER
# ─────────────────────────────────────────────────────────────────────────────

def _record_audio_vad(device: str = None) -> np.ndarray:
    """
    Stream audio in 100 ms chunks via arecord with real-time VAD.

    - Discards all silence before speech starts (no dead air sent to STT).
    - Starts collecting once RMS crosses VAD_SPEECH_THRESH.
    - Stops collecting VAD_SILENCE_AFTER seconds after speech drops below
      VAD_SILENCE_THRESH, or after VAD_MAX_DURATION seconds (hard cap).
    - Returns empty array if no speech detected or utterance is too short.

    Result: response arrives as soon as the user stops talking, not after
    a fixed 5-second wait.
    """
    global _working_arecord_dev

    primary = device or _mic_device
    card    = _mic_card_index
    devices = [primary, f"hw:{card},0", f"plughw:{card},0", "default"]
    seen    = set()
    devices = [d for d in devices if not (d in seen or seen.add(d))]

    # Probe once, cache so subsequent calls are instant
    if _working_arecord_dev:
        working_dev = _working_arecord_dev
    else:
        working_dev = None
        for dev in devices:
            test_file = tempfile.mktemp(suffix=".wav")
            try:
                r = subprocess.run(
                    ["arecord", "-D", dev, "-f", "S16_LE",
                     "-r", str(MIC_RATE), "-c", "1", "-d", "1", test_file],
                    capture_output=True, timeout=4
                )
                if r.returncode == 0:
                    working_dev          = dev
                    _working_arecord_dev = dev
                    print(f"🎤 VAD using device: {dev}")
                    break
            except Exception:
                pass
            finally:
                try:
                    os.remove(test_file)
                except Exception:
                    pass

    if not working_dev:
        print("❌ arecord failed on all devices — run: arecord -l  to check mic")
        return np.array([], dtype=np.float32)

    bytes_per_chunk  = int(MIC_RATE * _VAD_CHUNK_SECS) * 2   # 16-bit = 2 bytes/sample
    speech_chunks    = []
    speech_started   = False
    silence_duration = 0.0
    speech_duration  = 0.0
    total_duration   = 0.0

    # Launch arecord in streaming mode — no -d limit, killed when done
    proc = subprocess.Popen(
        ["arecord", "-D", working_dev, "-f", "S16_LE",
         "-r", str(MIC_RATE), "-c", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    try:
        while total_duration < VAD_MAX_DURATION:
            raw = proc.stdout.read(bytes_per_chunk)
            if not raw or len(raw) < bytes_per_chunk:
                break

            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            chunk = np.clip(chunk * MIC_GAIN, -1.0, 1.0)
            rms   = float(np.sqrt(np.mean(chunk ** 2)))
            total_duration += _VAD_CHUNK_SECS

            if rms >= VAD_SPEECH_THRESH:
                if not speech_started:
                    print("🎙️ Speech detected")
                    speech_started   = True
                silence_duration  = 0.0
                speech_duration  += _VAD_CHUNK_SECS
                speech_chunks.append(chunk)

            elif speech_started:
                # Keep trailing silence so end-of-word isn't clipped
                silence_duration += _VAD_CHUNK_SECS
                speech_chunks.append(chunk)
                if silence_duration >= VAD_SILENCE_AFTER:
                    print(f"🔇 End of speech ({speech_duration:.1f}s captured)")
                    break
            # Before speech: discard silently — don't accumulate dead air
    finally:
        proc.kill()
        proc.wait()

    if not speech_started or speech_duration < VAD_MIN_SPEECH:
        return np.array([], dtype=np.float32)

    return np.concatenate(speech_chunks)


# ─────────────────────────────────────────────────────────────────────────────
#  WEBSOCKET STT
# ─────────────────────────────────────────────────────────────────────────────

# Max seconds of speech allowed for wake word detection (short utterance only)
_WAKE_MAX_SPEECH = 1.5


async def _ws_listen_once(max_speech_secs: float = None) -> str:
    """Record via streaming VAD, resample to 16 kHz, send to STT server.
    If max_speech_secs is set, discard and return '' if audio is longer."""
    uri = f"ws://{STT_SERVER_IP}:{STT_PORT}"
    try:
        loop  = asyncio.get_event_loop()
        audio = await loop.run_in_executor(None, _record_audio_vad)

        if audio.size == 0:
            return ""

        # Reject audio that's too long to be a wake word
        duration = audio.size / MIC_RATE
        if max_speech_secs and duration > max_speech_secs:
            return ""

        audio_16k = resample_poly(audio, TARGET_RATE, MIC_RATE).astype(np.float32)

        async with _websockets_lib.connect(uri) as ws:
            await ws.send(audio_16k.tobytes())
            text = await ws.recv()
            return text.strip() if text else ""
    except Exception as e:
        print(f"❌ WebSocket STT error: {e}")
        return ""


# Persistent per-thread event loop — avoids ~20 ms loop-creation overhead
_listen_loop: asyncio.AbstractEventLoop | None = None


def listen(max_speech_secs: float = None) -> str:
    """
    Blocking wrapper — safe to call from any thread.
    Reuses a persistent event loop per thread; recreates if closed.
    """
    if not _websockets_available:
        return ""
    global _listen_loop
    try:
        if _listen_loop is None or _listen_loop.is_closed():
            _listen_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_listen_loop)
        return _listen_loop.run_until_complete(_ws_listen_once(max_speech_secs))
    except Exception as e:
        print(f"❌ listen() error: {e}")
        _listen_loop = None
        return ""


# ─────────────────────────────────────────────────────────────────────────────
#  ALSA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _scan_alsa_cards(tool: str) -> list[tuple[str, str, str]]:
    results = []
    try:
        r = subprocess.run([tool, "-l"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.startswith("card"):
                try:
                    card_num  = line.split(":")[0].strip().split()[-1]
                    short     = line.split(":")[1].strip().split("[")[0].strip()
                    card_name = short.replace(" ", "_")
                    dev_num   = line.split("device")[-1].strip().split(":")[0].strip()
                    results.append((card_num, card_name, dev_num))
                except Exception:
                    pass
    except Exception as e:
        print(f"⚠️ {tool} -l error: {e}")
    return results


def _is_bluetooth_connected() -> bool:
    try:
        r = subprocess.run(
            ["bluetoothctl", "devices", "Connected"],
            capture_output=True, text=True, timeout=2
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


def find_usb_audio_device() -> tuple[str, str]:
    """Return (aplay_device_str, card_num). Priority: 3.5mm → BT → USB → HDMI → fallback."""
    cards = _scan_alsa_cards("aplay")

    for card_num, card_name, dev_num in cards:
        if "bcm2835" in card_name.lower() or "headphone" in card_name.lower():
            try:
                r = subprocess.run(
                    ["amixer", "-c", card_num, "get", "Headphone"],
                    capture_output=True, text=True, timeout=2
                )
                if "Front Left:" in r.stdout or "Front Right:" in r.stdout:
                    print(f"🔊 3.5mm jack: card {card_num} ({card_name})")
                    return f"plughw:{card_num},{dev_num}", card_num
                print("⚠️ 3.5mm exists but nothing plugged — checking Bluetooth...")
            except Exception:
                print("⚠️ 3.5mm detection failed — checking Bluetooth...")

    if _is_bluetooth_connected():
        for card_num, card_name, dev_num in cards:
            if any(k in card_name.lower() for k in ("blue", "bt", "a2dp")):
                print(f"🔊 Bluetooth: card {card_num} ({card_name})")
                return f"plughw:{card_num},{dev_num}", card_num

    for card_num, card_name, dev_num in cards:
        if any(k in card_name.lower() for k in ("usb", "uac", "pnp")):
            print(f"🔊 USB: card {card_num} ({card_name})")
            return f"plughw:{card_num},{dev_num}", card_num

    for card_num, card_name, dev_num in cards:
        if any(k in card_name.lower() for k in ("hdmi", "vc4")):
            print(f"🔊 HDMI: card {card_num} ({card_name})")
            return f"plughw:{card_num},{dev_num}", card_num

    for card_num, card_name, dev_num in cards:
        if "bcm2835" in card_name.lower():
            print(f"🔊 Fallback bcm2835: card {card_num}")
            return f"plughw:{card_num},{dev_num}", card_num

    print("⚠️ No audio device — defaulting to plughw:0,0")
    return "plughw:0,0", "0"


def find_usb_mic_device() -> tuple[str, str, str]:
    cards = _scan_alsa_cards("arecord")
    if cards:
        card_num, card_name, dev_num = cards[0]
        print(f"🎤 USB mic: card {card_num} ({card_name}), device {dev_num}")
        return f"plughw:{card_num},{dev_num}", card_num, dev_num
    print("⚠️ No USB capture device — using fallback plughw:3,0")
    return "plughw:3,0", "3", "0"


def _aplay_with_retry(device: str, filepath: str, retries: int = 3) -> None:
    for attempt in range(retries):
        if os.system(f"aplay -D {device} {filepath} 2>/dev/null") == 0:
            return
        if attempt < retries - 1:
            time.sleep(0.3)
    os.system(f"aplay {filepath} 2>/dev/null")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class BuddyPi:
    """Pi Hardware Service — connects to remote brain via HTTP API."""

    _WAKE_WORDS = ["hey buddy", "hi buddy", "okay buddy", "ok buddy", "buddy"]

    _DIR_MAP = {
        "forward":  "F", "ahead": "F", "front": "F",
        "backward": "B", "back":  "B", "reverse": "B",
        "left":     "L",
        "right":    "R",
    }

    _MOTOR_CONFIRMATIONS = {
        "F": "Moving forward.",
        "B": "Moving backward.",
        "L": "Turning left.",
        "R": "Turning right.",
        "S": "Stopped.",
    }

    _SLEEP_PHRASES = [
        "go to sleep", "sleep now", "goodbye",
        "bye buddy", "shut down", "sleep"
    ]
    _SLEEP_REPLIES = ["Okay, going to sleep!", "Night night!", "See you later!"]
    _THINK_SOUNDS  = ["Hmm", "Let me think", "Umm", "One sec"]

    def __init__(self, config: Optional[Config] = None,
                 llm_service_url: Optional[str] = None):
        self.config          = config or Config.from_env()
        self.llm_service_url = llm_service_url or self.config.llm_service_url
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # ── Audio ─────────────────────────────────────────────────────────────
        self.aplay_device, self.usb_card_index = find_usb_audio_device()
        print(f"🔊 Playback device : {self.aplay_device}")

        self.arecord_device, self.mic_card_index, self.mic_dev_index = \
            find_usb_mic_device()
        print(f"🎤 Capture device  : {self.arecord_device}")

        global _mic_card_index, _mic_device
        _mic_card_index = self.mic_card_index
        _mic_device     = self.arecord_device

        # ── Camera ────────────────────────────────────────────────────────────
        self._init_camera()
        self.csi_process = None

        # ── Vision ────────────────────────────────────────────────────────────
        self.detector        = FaceDetector(self.config.cascade_path, self.config)
        self.face_recognizer = FaceRecognizer(self.config.model_path, self.config)
        self.stability       = StabilityTracker(self.config)

        print("🔍 Initializing object detection...")
        self.object_detector    = ObjectDetector(confidence_threshold=0.6)
        self.stable_objects     = set()
        self.persistent_objects = set()
        print("✅ Object detection ready")

        # ── Servo ─────────────────────────────────────────────────────────────
        # try:
        #     self.servo             = ServoController()
        #     self.servo_enabled     = True
        #     self._servo_looking_up = False
        # except Exception as e:
        #     print(f"⚠️ Servo init failed: {e}")
        self.servo             = None
        self.servo_enabled     = False
        self._servo_looking_up = False

        # ── Motors ────────────────────────────────────────────────────────────
        self.motors = MotorController(port=ARDUINO_PORT, baud=ARDUINO_BAUD)
        self.motors.set_obstacle_callback(self._on_obstacle)
        self.motors.set_clear_callback(self._on_clear)
        self._obstacle_detected = False

        # ── Speech ────────────────────────────────────────────────────────────
        self._init_speech()

        # ── State ─────────────────────────────────────────────────────────────
        self.last_recognition_time      = 0
        self.last_object_detection_time = 0
        self.last_object_clear_time     = time.time()
        self.current_detections         = []
        self.running      = False
        self.is_speaking  = False
        self.active_user  = None
        self.sleep_mode   = False
        self.frame_count  = 0
        self.last_frame   = None

        self.recognition_attempts  = {}
        self.max_attempts          = 5
        self.recognition_threshold = 0.4
        self.face_lost_count       = 0
        self.max_face_lost         = 10

        self._notif_lock     = threading.Lock()
        self._notif_queue    = []
        self._thinking_token = None

        print("🤖 Buddy Pi Hardware Ready")
        self._start_phone_listener()

    # ─────────────────────────────────────────────────────────────────────────

    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=self.config.log_format
        )

    # ── Camera ───────────────────────────────────────────────────────────────

    def _init_camera(self):
        self.csi_frame_path = "/tmp/csi_frame.jpg"
        self.use_csi        = False
        self.cap            = None
        self.csi_process    = None

        if shutil.which("rpicam-hello"):
            print("📷 Starting CSI camera helper...")
            try:
                self.csi_process = subprocess.Popen(
                    ["/usr/bin/python3", "csi_camera_helper.py"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                for _ in range(50):
                    if os.path.exists(self.csi_frame_path):
                        self.use_csi = True
                        print("✅ CSI camera ready")
                        atexit.register(self._cleanup_csi)
                        return
                    time.sleep(0.1)
                print("⚠️ CSI frame not received — stopping helper")
                self.csi_process.terminate()
                self.csi_process = None
            except Exception as e:
                print(f"⚠️ CSI camera failed: {e}")
                if self.csi_process:
                    self.csi_process.terminate()
                    self.csi_process = None

        print("📷 Searching for USB camera...")
        for idx in (0, 1, 2):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.config.camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,   self.config.camera_buffer_size)
                cap.set(cv2.CAP_PROP_FPS,          self.config.camera_fps)
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    print(f"✅ USB camera at /dev/video{idx}")
                    return
            cap.release()

        raise RuntimeError("❌ No camera found")

    def _read_frame(self):
        if self.use_csi and os.path.exists(self.csi_frame_path):
            frame = cv2.imread(self.csi_frame_path)
            return (True, frame) if frame is not None else (False, None)
        if self.cap:
            return self.cap.read()
        return False, None

    # ── Speech ───────────────────────────────────────────────────────────────

    def _init_speech(self):
        if not _websockets_available:
            print("❌ WebSocket STT not available — speech disabled")
            self.speech_enabled = False
            return

        self.tts_voice      = "en-IN-NeerjaNeural"
        self.speech_enabled = True

        try:
            os.environ["SDL_AUDIODRIVER"] = "alsa"
            os.environ["AUDIODEV"]        = f"plughw:{self.usb_card_index},0"
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        except Exception as e:
            print(f"⚠️ pygame mixer init failed (non-fatal): {e}")

        print(f"✅ Speech ready  (VAD STT @ {STT_SERVER_IP}:{STT_PORT} | TTS card {self.usb_card_index})")

    # ── Alexa-style beep ─────────────────────────────────────────────────────

    def _generate_beep_wav(self):
        """Generate beep WAV once and cache it at /tmp/buddy_beep.wav."""
        import wave, struct
        wav_path = "/tmp/buddy_beep.wav"
        if os.path.exists(wav_path):
            return wav_path
        sample_rate = 22050
        frames = []
        for freq, dur in ((880, 0.12), (1100, 0.15)):
            n = int(sample_rate * dur)
            for i in range(n):
                val = int(14000 * np.sin(2 * np.pi * freq * i / sample_rate))
                frames.append(struct.pack('<h', val))
        with wave.open(wav_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
        return wav_path

    def _play_listen_beep(self):
        """Two-tone rising chime via aplay. Tries multiple devices until one works."""
        try:
            wav_path = self._generate_beep_wav()
            devices  = [self.aplay_device, "plughw:0,0", "default"]
            for dev in devices:
                r = subprocess.run(
                    ["aplay", "-D", dev, "-q", wav_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                if r.returncode == 0:
                    return
            print("⚠️ Beep: no working audio device found")
        except Exception as e:
            print(f"⚠️ Beep failed: {e}")

    # ── TTS ──────────────────────────────────────────────────────────────────

    def speak(self, text: str):
        """Non-blocking TTS. Spawns a daemon thread with its own event loop."""
        if not self.speech_enabled or not text:
            return
        self.is_speaking = True  # set before thread starts to avoid race

        def _speak():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._generate_and_play_speech(text))
                finally:
                    loop.close()
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                self.is_speaking = False

        threading.Thread(target=_speak, daemon=True).start()

    async def _generate_and_play_speech(self, text: str):
        """Stream TTS MP3 directly into mpg123 (lowest latency).
        Falls back to ffmpeg+aplay if mpg123 is absent."""
        try:
            communicate = edge_tts.Communicate(text, self.tts_voice)

            if shutil.which("mpg123"):
                proc = subprocess.Popen(
                    ["mpg123", "-q", "-a", f"hw:{self.usb_card_index},0", "-"],
                    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        proc.stdin.write(chunk["data"])
                proc.stdin.close()
                proc.wait()
                return

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

            if shutil.which("ffmpeg"):
                if os.system(
                    f"ffmpeg -y -loglevel quiet -i {mp3_path}"
                    f" -ar 22050 -ac 2 -sample_fmt s16 {wav_path}"
                ) == 0:
                    _aplay_with_retry(self.aplay_device, wav_path)
                    return

            print("⚠️ mpg123 and ffmpeg not found — run: sudo apt install mpg123")
            _aplay_with_retry(self.aplay_device, mp3_path)

        except Exception as e:
            print(f"Edge TTS Error: {e}")

    def _wait_for_tts(self, timeout: float = 15.0):
        """Block until is_speaking clears, then add a short ALSA-release buffer."""
        t0 = time.time()
        while self.is_speaking and (time.time() - t0) < timeout:
            time.sleep(0.05)
        time.sleep(0.3)

    # ── Thinking sound ───────────────────────────────────────────────────────

    def _play_thinking_sound(self):
        """Fire a filler sound only if the brain call takes > 1.5 s."""
        import uuid
        token = str(uuid.uuid4())
        self._thinking_token = token

        def _delayed(t):
            time.sleep(1.5)
            if getattr(self, "_thinking_token", None) == t:
                self.speak(random.choice(self._THINK_SOUNDS))

        threading.Thread(target=_delayed, args=(token,), daemon=True).start()

    # ── Listening ────────────────────────────────────────────────────────────

    def listen_for_speech(self) -> str:
        """Record via streaming VAD → STT → return transcript, or '' on silence."""
        if not self.speech_enabled or not _websockets_available:
            return ""
        print("🎤 Listening (VAD)...")
        text = listen()
        if text and len(text) > 2:
            print(f"🎤 Heard: '{text}'")
            return text
        print("🔇 No speech detected")
        return ""

    def _is_wake_word(self, text: str) -> bool:
        t = text.lower().strip()
        if len(t.split()) > 4:
            return False
        return any(w in t for w in self._WAKE_WORDS)

    def _wake_word_loop(self):
        print("👂 Waiting for 'Buddy'...")
        while self.running and not self.sleep_mode:
            if _vosk_available:
                detected = _vosk_listen_for_wake_word(self.arecord_device, self._WAKE_WORDS)
            else:
                text     = listen(_WAKE_MAX_SPEECH) if _websockets_available else ""
                detected = bool(text) and self._is_wake_word(text)

            if not detected:
                continue
            print("✅ Wake word detected")
            self._play_listen_beep()
            user_text = self.listen_for_speech()
            if user_text:
                self._process_input(user_text)
            self._wait_for_tts()
            with self._notif_lock:
                has_notifs = bool(self._notif_queue)
            if has_notifs:
                threading.Thread(target=self._flush_notifications, daemon=True).start()
            print("👂 Back to idle...")

    # ── Brain service ─────────────────────────────────────────────────────────

    def _call_brain_service(self, user_input: str,
                            recognized_user: Optional[str] = None) -> dict:
        try:
            user_lower = user_input.lower()

            identity_patterns = [
                'who am i', 'who is this', 'who is here', 'who do you see',
                'recognize me', 'recognize this', 'can you see me', 'do you know me',
                'identify me', 'identify this', 'scan face', 'scan me'
            ]
            if any(p in user_lower for p in identity_patterns):
                recognized_user = self._force_face_recognition()
                if recognized_user:
                    print(f"🔍 Forced recognition: {recognized_user}")

            objects = []
            if any(p in user_lower for p in
                   ['what is', 'what do you see', 'in my hand', 'holding', 'show you']):
                ret, frame = self._read_frame()
                if ret:
                    fresh = self.object_detector.detect(frame)
                    if fresh:
                        objects = [d['name'] for d in fresh]
                        self.persistent_objects.update(objects)
                        print(f"🔄 Fresh detection: {objects}")

            if not objects:
                objects = (list(self.persistent_objects) if self.persistent_objects
                           else list(self.stable_objects))

            response = requests.post(
                f"{self.llm_service_url}/chat",
                json={
                    "user_input":      user_input,
                    "recognized_user": recognized_user,
                    "objects_visible": objects,
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            raise Exception(f"Brain service error: {response.status_code}")

        except Exception as e:
            self.logger.error(f"Brain service call failed: {e}")
            return {
                "reply":        "Sorry, I'm having trouble thinking right now.",
                "intent":       "conversation",
                "emotion":      "apologetic",
                "raw_response": ""
            }

    # ── Input processing ──────────────────────────────────────────────────────

    def _process_input(self, user_text: str):
        """
        Priority:
          1. STOP   — instant, no brain
          2. MOVE   — instant, no brain
          3. SLEEP  — no brain
          4. Everything else → brain (deferred thinking sound)
        """
        try:
            t = user_text.lower()

            # 1. Stop
            if self._is_stop_command(t):
                self._execute_movement("S")
                self.speak("Stopped.")
                return

            # 2. Move
            parsed = self._parse_movement_intent(t)
            if parsed is not None:
                cmd, duration = parsed
                print(f"[MOTOR] {cmd!r} duration={duration}")
                self._execute_movement(cmd, duration)
                self.speak(self._MOTOR_CONFIRMATIONS.get(cmd, "On it."))
                return

            # 3. Sleep
            if any(phrase in t for phrase in self._SLEEP_PHRASES):
                self.speak(random.choice(self._SLEEP_REPLIES))
                time.sleep(1.5)
                self._enter_sleep_mode()
                return

            # 4. Name registration
            if any(phrase in t for phrase in ('my name is', 'i am', "i'm")):
                name = self._extract_name(user_text)
                if name and self.active_user != name and self.last_frame is not None:
                    faces = self.detector.detect(self.last_frame)
                    if faces:
                        x, y, w, h = self.detector.get_largest_face(faces)
                        if w > 80 and h > 80:
                            face_roi = self.last_frame[y:y+h, x:x+w]
                            if self.face_recognizer.add_face(name, face_roi):
                                self.active_user = name
                                print(f"Registered new face: {name}")

            # 5. Brain
            self._play_thinking_sound()
            response = self._call_brain_service(user_text, self.active_user)
            self._thinking_token = None
            self._display_response(response)

        except Exception as e:
            self._thinking_token = None
            print(f"Error processing input: {e}")

    def _is_stop_command(self, text: str) -> bool:
        return any(w in text for w in
                   ("stop", "halt", "freeze", "don't move", "do not move"))

    def _parse_movement_intent(self, text: str):
        cmd = None
        for word, c in self._DIR_MAP.items():
            if word in text:
                cmd = c
                break
        if cmd is None:
            return None

        duration = None
        m = re.search(r"for\s+(\d+(?:\.\d+)?)\s*(second|sec|minute|min)", text)
        if m:
            val  = float(m.group(1))
            unit = m.group(2)
            duration = val * 60 if unit.startswith("min") else val

        return (cmd, duration)

    def _extract_name(self, text: str) -> str:
        """Extract a single-word name. Guards against common non-name words."""
        _COMMON = {
            "going", "here", "back", "home", "fine", "good", "okay",
            "ready", "not", "sure", "just", "also", "well", "sorry"
        }
        text = text.lower()
        candidate = ""
        if 'my name is' in text:
            candidate = text.split('my name is')[1].strip().split()[0]
        elif "i'm" in text:
            candidate = text.split("i'm")[1].strip().split()[0]
        elif 'i am' in text:
            candidate = text.split('i am')[1].strip().split()[0]

        candidate = candidate.strip(".,!?").title()
        if (candidate
                and candidate.lower() not in _COMMON
                and candidate.isalpha()
                and len(candidate) > 1):
            return candidate
        return ""

    # ── Display / respond ─────────────────────────────────────────────────────

    def _display_response(self, response: dict):
        if not response:
            return

        reply        = response.get("reply", "")
        intent       = response.get("intent", "conversation")
        raw_response = response.get("raw_response", "")

        print(f"\nBuddy: {raw_response if raw_response else reply}")

        if intent != "conversation":
            print(f"[INTENT: {intent}]")
            _brain_move = {
                "move_forward":  "F", "move_backward": "B",
                "move_left":     "L", "move_right":    "R",
                "stop":          "S",
            }
            if intent in _brain_move:
                cmd      = _brain_move[intent]
                duration = None if cmd == "S" else 1.5
                self._execute_movement(cmd, duration)

        self.speak(reply)

        emotion = response.get("emotion", "")
        if emotion:
            print(f"🎭 Emotion: '{emotion}'")
            if self.motors.is_connected():
                threading.Timer(0.3, lambda e=emotion: self.motors.emotion_move(e)).start()

    # ── Motor control ─────────────────────────────────────────────────────────

    def _execute_movement(self, cmd: str, duration: float = None):
        if not self.motors.is_connected():
            print(f"⚠️ Motors not connected — ignoring: {cmd}")
            return
        if cmd == "S":
            self.motors.stop()
        else:
            self.motors.move(cmd, duration)

    def _on_obstacle(self):
        print("🚧 Obstacle detected")
        self._obstacle_detected = True
        if not self.is_speaking:
            self.speak("Obstacle detected. I've stopped.")

    def _on_clear(self):
        print("✅ Path clear")
        self._obstacle_detected = False
        if not self.is_speaking:
            self.speak("Path is clear.")

    # ── Face recognition ──────────────────────────────────────────────────────

    def _force_face_recognition(self) -> Optional[str]:
        try:
            ret, frame = self._read_frame()
            if not ret:
                return None
            faces = self.detector.detect(frame)
            if not faces:
                return None
            recognized = []
            for (x, y, w, h) in faces:
                if w > 80 and h > 80:
                    try:
                        name, conf = self.face_recognizer.recognize(
                            frame[y:y+h, x:x+w]
                        )
                        if name != "Unknown" and conf > self.recognition_threshold:
                            recognized.append(name)
                            print(f"✅ Recognised: {name} ({conf:.2f})")
                    except Exception as e:
                        print(f"Face recognition error: {e}")
            if recognized:
                self.active_user = recognized[0]
                return recognized[0]
            return None
        except Exception as e:
            print(f"Force recognition error: {e}")
            return None

    def _greet_recognized_user(self, name: str):
        self.speak(f"Hey {name}, good to see you!")

    # ── Frame processing ──────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> tuple:
        try:
            faces = self.detector.detect(frame)
        except Exception as e:
            print(f"Face detection error: {e}")
            faces = []

        name          = None
        confidence    = 0.0
        face_detected = len(faces) > 0

        if face_detected:
            largest   = self.detector.get_largest_face(faces)
            is_stable = self.stability.update(largest)
            now       = time.time()

            if is_stable and (now - self.last_recognition_time) > 30.0:
                x, y, w, h = largest
                if w > self.config.min_face_size[0] and h > self.config.min_face_size[1]:
                    name, confidence = self.face_recognizer.recognize(
                        frame[y:y+h, x:x+w]
                    )
                    print(f"🔍 Recognition → {name} ({confidence:.2f})")
                    if name != "Unknown" and confidence > self.recognition_threshold:
                        if self.active_user != name:
                            print(f"✅ New user: {name}")
                            self.active_user = name
                            threading.Thread(
                                target=self._greet_recognized_user,
                                args=(name,), daemon=True
                            ).start()
                    else:
                        print(f"❓ Not recognised (conf={confidence:.2f})")
                    self.last_recognition_time = now
        else:
            self.stability.reset()

        now = time.time()
        if (now - self.last_object_detection_time) > 5.0:
            self.current_detections         = self._process_objects(frame)
            self.last_object_detection_time = now

        if (now - self.last_object_clear_time) > 60.0:
            print(f"🧹 Clearing old objects: {self.persistent_objects}")
            self.persistent_objects.clear()
            self.last_object_clear_time = now

        processed = self._draw_visualization(
            frame, faces, name, confidence, self.current_detections
        )

        # if self.servo_enabled and self.servo:
        #     if face_detected and self._servo_looking_up:
        #         self.servo.look_center(smooth=True)
        #         self._servo_looking_up = False
        #     elif not face_detected and not self._servo_looking_up:
        #         self.servo.look_up(smooth=True)
        #         self._servo_looking_up = True

        return processed, face_detected, name, confidence

    def _process_objects(self, frame):
        try:
            detections = self.object_detector.detect(frame)
            if detections:
                current = set(d['name'] for d in detections)
                print(f"🔍 Objects: {[(d['name'], d['confidence']) for d in detections]}")
                self.stable_objects = current
                self.persistent_objects.update(current)
                return detections
            self.stable_objects = set()
            return []
        except Exception as e:
            print(f"⚠️ Object detection error: {e}")
            self.stable_objects = set()
            return []

    def _draw_visualization(self, frame, faces, name=None,
                             confidence=0.0, detections=None):
        if detections:
            frame = self.object_detector.draw_detections(frame, detections)

        for (x, y, w, h) in faces:
            color = (0, 255, 0) if self.stability.is_stable else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            if name and name != "Unknown":
                cv2.putText(frame, f"{name} ({confidence:.0%})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        status = (f"💬 {self.active_user}" if self.active_user
                  else "🔍 Looking for people")
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if self.active_user else (0, 255, 255), 2)

        if self.persistent_objects:
            cv2.putText(frame,
                        f"Objects: {', '.join(list(self.persistent_objects)[:3])}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        if self.stable_objects:
            cv2.putText(frame,
                        f"Current: {', '.join(list(self.stable_objects)[:2])}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        return frame

    # ── Startup greeting ──────────────────────────────────────────────────────

    def _startup_greeting(self):
        time.sleep(1.0)
        self.speak("Hey! I'm Buddy. Call me anytime.")

    # ── Sleep / wake ──────────────────────────────────────────────────────────

    def _enter_sleep_mode(self):
        print("\n😴 Entering sleep mode...")
        self.sleep_mode = True

    def _wake_up_and_restart(self):
        print("😊 Waking up!")
        self.sleep_mode = False
        self._init_speech()
        self.speak("I'm awake! Say buddy to talk to me.")

    # ── Phone notifications ───────────────────────────────────────────────────

    def _start_phone_listener(self):
        import json
        from http.server import BaseHTTPRequestHandler, HTTPServer
        buddy = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get('Content-Length', 0))
                body   = self.rfile.read(length)
                try:
                    data   = json.loads(body)
                    result = process_notification(
                        data.get('app', 'Unknown'),
                        data.get('title', ''),
                        data.get('message', ''),
                    )
                    if result['status'] == 'received':
                        threading.Thread(
                            target=buddy._on_phone_notification,
                            args=(result,), daemon=True
                        ).start()
                    resp = json.dumps(result).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(resp)
                except Exception:
                    self.send_response(400)
                    self.end_headers()

            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"message":"BUDDY phone link active"}')

            def log_message(self, *args):
                pass

        def _run():
            HTTPServer(('0.0.0.0', 8001), _Handler).serve_forever()

        threading.Thread(target=_run, daemon=True).start()
        print('📱 Phone notification listener started on port 8001')

    def _on_phone_notification(self, notif: dict):
        if notif.get("decision") == "ignore" or self.sleep_mode:
            return
        with self._notif_lock:
            self._notif_queue.append(notif)
        if not self.is_speaking:
            threading.Thread(target=self._flush_notifications, daemon=True).start()

    def _flush_notifications(self):
        while True:
            with self._notif_lock:
                if not self._notif_queue:
                    break
                notif = self._notif_queue.pop(0)
            self._wait_for_tts()
            app     = notif.get("app", "someone")
            sender  = notif.get("sender", "")
            message = notif.get("message", "")
            urgency = "urgently" if notif.get("decision") == "important" else "casually"
            prompt  = (
                f"You just noticed a notification on the user's phone. "
                f"App: {app}. From: {sender}. Message: '{message}'. "
                f"Tell the user about it {urgency}, like a friend who glanced at their "
                f"phone — natural, short, no robotic phrasing."
            )
            print(f"📱 Notification from {app} ({sender}): {message}")
            response = self._call_brain_service(prompt, recognized_user=self.active_user)
            if response and response.get("reply"):
                self.speak(response["reply"])
                self._wait_for_tts()

    # ── Main loops ────────────────────────────────────────────────────────────

    def run(self):
        self.running = True
        try:
            self._conversation_loop()
        except KeyboardInterrupt:
            print("\n⚠️ Keyboard interrupt")
            self.running = False
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
        finally:
            self.cleanup()

    def _conversation_loop(self):
        """
        Camera loop  → background thread (never blocks)
        Greeting     → background thread (never blocks)
        Wake-word    → main thread (blocks here until sleep/shutdown)
        """
        print("🎯 Starting conversation loop")

        def _camera_loop():
            while self.running and not self.sleep_mode:
                ret, frame = self._read_frame()
                if not ret:
                    continue
                self.last_frame   = frame
                self.frame_count += 1
                if self.frame_count % 5 == 0:
                    processed, _, _, _ = self._process_frame(frame)
                else:
                    processed = frame
                cv2.imshow('Buddy Vision', processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                time.sleep(0.033)

        threading.Thread(target=_camera_loop, daemon=True).start()
        threading.Thread(target=self._startup_greeting, daemon=True).start()

        time.sleep(0.5)  # head-start so greeting speaks before wake-word loop

        self._wake_word_loop()

        if self.sleep_mode and self.running:
            self._sleep_loop()

    def _sleep_loop(self):
        print("😴 Starting sleep loop")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        self.speak("Going to sleep. Say 'Hey Buddy' to wake me up.")
        time.sleep(2)
        print("😴 Listening for wake word...")

        while self.running and self.sleep_mode:
            try:
                if _vosk_available:
                    if _vosk_listen_for_wake_word(self.arecord_device, self._WAKE_WORDS):
                        print("✅ [SLEEP] Wake word detected")
                        self._wake_up_and_restart()
                        break
                else:
                    text = listen(_WAKE_MAX_SPEECH) if _websockets_available else ""
                    if text and self._is_wake_word(text):
                        self._wake_up_and_restart()
                        break

            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(f"❌ [SLEEP] Error: {e}")
                time.sleep(1.0)

        if not self.sleep_mode and self.running:
            self._conversation_loop()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _cleanup_csi(self):
        if hasattr(self, 'csi_process') and self.csi_process:
            self.csi_process.terminate()

    def cleanup(self):
        self.running = False

        # if self.servo_enabled and self.servo:
        #     try:
        #         self.servo.look_center(smooth=False)
        #         time.sleep(0.3)
        #         self.servo.cleanup()
        #     except Exception:
        #         pass

        if hasattr(self, 'motors'):
            try:
                self.motors.stop()
                self.motors.cleanup()
            except Exception:
                pass

        try:
            requests.post(f"{self.llm_service_url}/clear", timeout=5)
        except Exception:
            pass

        if self.use_csi and hasattr(self, 'csi_process') and self.csi_process:
            self.csi_process.terminate()

        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

        try:
            cv2.destroyAllWindows()
            pygame.mixer.quit()
        except Exception:
            pass

        print("\nBuddy Pi: Goodbye!")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    try:
        config  = Config.from_env()
        llm_url = config.llm_service_url

        print(f"🤖 Starting Buddy Pi...")
        print(f"🔗 LLM Service : {llm_url}")
        print(f"📷 Camera      : {config.camera_index}")

        buddy = BuddyPi(config)

        def shutdown(sig, frame):
            print("\n⚠️ Shutting down...")
            buddy.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT,  shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        buddy.run()
        return 0

    except Exception as e:
        logging.error(f"Startup failed: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())