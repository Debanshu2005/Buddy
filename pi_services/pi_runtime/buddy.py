"""
Buddy Pi — main driver.

Wires together all mixin modules into a single BuddyPi class.
Run from pi_services/:
    python pi_runtime/buddy.py
"""
from __future__ import annotations

import atexit
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

# ── PC auto-discovery ─────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = ROOT_DIR / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
        print(f"[Config] Loaded .env from {_env_path}")
except ImportError:
    pass

try:
    from core.pc_discovery import discover_pc_ip
    _pc_ip = discover_pc_ip(verbose=True)
    os.environ.setdefault("BUDDY_STT_SERVER_IP", _pc_ip)
    os.environ.setdefault("BUDDY_PC_CAMERA_IP", _pc_ip)
    os.environ.setdefault("LLM_SERVICE_URL", f"http://{_pc_ip}:8000")
except Exception as _e:
    print(f"[Discovery] Failed: {_e}")

# ── Vosk ──────────────────────────────────────────────────────────────────────
_VOSK_MODEL_PATH = str(ROOT_DIR / "vosk-model-small-en-us-0.15")
_vosk_model = None
_vosk_available = False
try:
    from vosk import Model as VoskModel, KaldiRecognizer as _KaldiRecognizer
    _vosk_model = VoskModel(_VOSK_MODEL_PATH)
    _vosk_available = True
    print("Vosk wake word model loaded")
except Exception as _e:
    print(f"Vosk not available: {_e} — falling back to STT wake word")

# ── Hardware / vision imports ─────────────────────────────────────────────────
try:
    from gpiozero import DistanceSensor
except Exception:
    DistanceSensor = None

from core.config import Config
from core.stability_tracker import StabilityTracker
from hardware.motor_controller import MotorController
from hardware.oled_eyes import OledEyes, EyeState
from phone_link.core import process_notification
from vision.behavior.pipeline import BehaviorDetectionPipeline, PipelineConfig
from vision.behavior.whatsapp_alert import send_whatsapp_alert as send_telegram_alert

# ── Mixin modules ─────────────────────────────────────────────────────────────
from modules.speech import SpeechMixin
from modules.wake import WakeMixin
from modules.conversation import ConversationMixin
from modules.camera import CameraMixin
from modules.safety import SafetyMixin
from modules.follow import FollowMixin
from modules.input_handler import InputHandlerMixin
from modules.emergency import EmergencyMixin
from modules.notifications import NotificationsMixin


# ── Settings ──────────────────────────────────────────────────────────────────
@dataclass
class RuntimeSettings:
    stt_server_ip: str = os.getenv("BUDDY_STT_SERVER_IP", "buddypc.local")
    stt_port: int = int(os.getenv("BUDDY_STT_PORT", "8765"))
    notification_port: int = int(os.getenv("BUDDY_NOTIFICATION_PORT", "8001"))
    arduino_port: str = os.getenv("BUDDY_ARDUINO_PORT", "").strip()
    arduino_baud: int = int(os.getenv("BUDDY_ARDUINO_BAUD", "115200"))
    use_servo: bool = True
    recognition_interval: float = 5.0
    object_interval_frames: int = 20
    face_interval_frames: int = 15
    display_enabled: bool = os.getenv("BUDDY_ENABLE_DISPLAY", "1") != "0"
    pc_camera_ip: str = os.getenv("BUDDY_PC_CAMERA_IP", "buddypc.local")
    pc_camera_port: int = 5000
    surveillance_enabled: bool = os.getenv("BUDDY_ENABLE_SURVEILLANCE", "0") == "1"
    surveillance_port: int = int(os.getenv("BUDDY_SURVEILLANCE_PORT", "8001"))
    surveillance_cooldown: float = float(os.getenv("BUDDY_SURVEILLANCE_COOLDOWN", "15.0"))
    ultrasonic_enabled: bool = os.getenv("BUDDY_ENABLE_ULTRASONIC", "0") == "1"
    ultrasonic_trigger_pin: int = int(os.getenv("BUDDY_ULTRASONIC_TRIGGER_PIN", "23"))
    ultrasonic_echo_pin: int = int(os.getenv("BUDDY_ULTRASONIC_ECHO_PIN", "24"))
    ultrasonic_stop_distance_m: float = float(os.getenv("BUDDY_ULTRASONIC_STOP_DISTANCE_M", "0.28"))
    ultrasonic_max_distance_m: float = float(os.getenv("BUDDY_ULTRASONIC_MAX_DISTANCE_M", "2.0"))


@dataclass
class _CameraTask:
    name: str
    runner_name: str
    interval_frames: int
    latest_frame: Optional[np.ndarray] = None
    latest_frame_count: int = 0
    busy: bool = False


# ── Surveillance client ───────────────────────────────────────────────────────
class _SurveillanceClient:
    POLL_INTERVAL = 0.5
    _CRITICAL = {"fall", "person_down", "hand_on_chest", "eyes_closed", "trembling", "head_tilt"}
    _HIGH = {"hands_raised", "prolonged_inactivity", "covering_face", "head_drooping"}
    _MEDIUM = {"clutching_stomach", "hunching", "crouching"}

    def __init__(self, pc_ip, port, alert_callback, cooldown=120.0):
        import requests as _req
        self._req = _req
        self._url = f"http://{pc_ip}:{port}/latest"
        self._alert_callback = alert_callback
        self._cooldown = cooldown
        self._last_fired: dict = {}
        self._seen_timestamps: set = set()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="SurveillanceClient").start()
        print(f"[Surveillance] Polling {self._url}")

    def stop(self):
        self._running = False

    def _can_fire(self, event_type):
        return (time.time() - self._last_fired.get(event_type, 0.0)) >= self._cooldown

    def _severity(self, event_type):
        if event_type in self._CRITICAL: return "critical"
        if event_type in self._HIGH: return "high"
        return "medium"

    def _loop(self):
        while self._running:
            try:
                resp = self._req.get(self._url, timeout=2)
                if resp.status_code == 200:
                    for event in resp.json().get("events", []):
                        etype = event.get("event_type", "")
                        if not etype: continue
                        ts = float(event.get("timestamp", 0.0))
                        if ts in self._seen_timestamps: continue
                        self._seen_timestamps.add(ts)
                        if len(self._seen_timestamps) > 500:
                            self._seen_timestamps = set(list(self._seen_timestamps)[-200:])
                        if not self._can_fire(etype): continue
                        self._last_fired[etype] = time.time()
                        self._alert_callback(
                            event_type=etype,
                            description=event.get("description", etype),
                            confidence=float(event.get("confidence", 0.0)),
                            severity=self._severity(etype),
                        )
            except Exception:
                pass
            time.sleep(self.POLL_INTERVAL)


# ── Main class ────────────────────────────────────────────────────────────────
class BuddyIntegratedPi(
    SpeechMixin,
    WakeMixin,
    ConversationMixin,
    CameraMixin,
    SafetyMixin,
    FollowMixin,
    InputHandlerMixin,
    EmergencyMixin,
    NotificationsMixin,
):
    # ── Class-level constants (shared across mixins) ──────────────────────────
    _IDENTITY_TRIGGERS = (
        "do you know me", "do u know me", "who am i", "do you recognize me",
        "do u recognize me", "recognize me", "identify me", "can you see me",
        "do you remember me", "do u remember me", "remember me", "have we met",
        "do you know who i am", "do u know who i am", "who is this",
    )
    _WAKE_WORDS = ("buddy", "hey buddy", "hi buddy", "buddy wake up", "hey buddy wake up", "wake up buddy")
    _SLEEP_PHRASES = ("go to sleep", "sleep now", "take a nap", "good night")
    _STOP_WORDS = ("stop", "halt", "freeze", "don't move", "do not move")
    _DIR_MAP = {
        "forward": "F", "ahead": "F", "backward": "B", "back": "B",
        "left": "L", "right": "R", "spin left": "L", "spin right": "R",
        "turn left": "L", "turn right": "R",
    }
    _MOTOR_CONFIRMATIONS = {"F": "Moving forward.", "B": "Moving backward.", "L": "Spinning left.", "R": "Spinning right.", "S": "Stopped."}
    _THINK_SOUNDS = ("Let me think.", "One moment.", "Thinking.")
    _RESPONSE_DELAY_MIN_SAMPLES = 5
    _RESPONSE_DELAY_RATIO = 2.0
    _RESPONSE_DELAY_SECONDS = 8.0
    _RESPONSE_AVG_ALPHA = 0.2
    _VOICE_WEAK_MIN_SAMPLES = 10
    _VOICE_WEAK_RATIO = 0.35
    _VOICE_AVG_ALPHA = 0.15
    _CONVERSATION_IDLE_TIMEOUT = 8.0
    _VISUAL_STILLNESS_SECONDS = 75.0
    _VISUAL_CHECK_COOLDOWN_SECONDS = 120.0
    _VISUAL_MOTION_THRESHOLD = 1.5
    _FOLLOW_TARGET_AREA_RATIO = 0.08
    _FOLLOW_TOO_CLOSE_AREA_RATIO = 0.22
    _FOLLOW_CENTER_TOLERANCE = 0.18
    _FOLLOW_COMMAND_INTERVAL = 0.35
    _FOLLOW_LOST_TIMEOUT = 2.5
    _FOLLOW_FACE_LOOK_UP_DEGREES = 30.0
    _FOLLOW_FACE_SCAN_SETTLE_SECONDS = 0.8
    _FOLLOW_FACE_SCAN_TIMEOUT = 1.8
    _ULTRASONIC_POLL_INTERVAL = 0.15
    _ULTRASONIC_ANNOUNCE_COOLDOWN = 3.0
    _SERVO_FACE_CENTER_TOLERANCE = 0.05
    _SERVO_FACE_GAIN = 48.0
    _SERVO_FACE_MIN_ANGLE = 30.0
    _SERVO_FACE_MAX_ANGLE = 150.0
    _SERVO_FACE_UPDATE_INTERVAL = 0.12
    _EYE_TRACK_UPDATE_INTERVAL = 0.08
    _BBOX_FALL_ASPECT_RATIO = 1.12
    _BBOX_FALL_MIN_AREA_RATIO = 0.055
    _BBOX_FALL_LOW_CENTER_RATIO = 0.54
    _BBOX_FALL_CONFIRM_SECONDS = 2.0
    _BBOX_FALL_ALERT_COOLDOWN_SECONDS = 120.0
    _BLOOD_MIN_REGION_RATIO = 0.008
    _BLOOD_MIN_SATURATION = 105
    _BLOOD_MIN_VALUE = 35
    _BLOOD_CONFIRM_SECONDS = 1.5
    _BLOOD_ALERT_COOLDOWN_SECONDS = 180.0
    _OK_RESPONSES = ("yes", "yeah", "yep", "ok", "okay", "fine", "i am fine", "i'm fine", "all good", "i am okay", "i'm okay", "yes i am", "yes i'm ok")
    _NOT_OK_RESPONSES = ("no", "nope", "not ok", "not okay", "help", "help me", "emergency", "call someone", "call family", "call emergency", "i need help", "i am not okay", "i'm not okay", "hurt", "pain")
    _EMERGENCY_PHRASES = (
        "emergency", "help me", "i need help", "call family", "call my family",
        "call someone", "call emergency", "call ambulance", "i fell", "i have fallen",
        "i am hurt", "i'm hurt", "chest pain", "can't breathe", "cannot breathe",
        "hard to breathe", "feel dizzy", "i feel dizzy", "i feel sick",
        "i am not feeling safe", "i'm not feeling safe", "i do not feel safe",
        "i don't feel safe", "i am unsafe", "i'm unsafe", "not safe", "unsafe",
        "i am not feeling well", "not feeling well", "i feel unwell",
        "i am in danger", "i'm in danger", "danger",
    )
    _EMOTION_MAP = {
        "happy": (EyeState.HAPPY, None, "W"), "joyful": (EyeState.HAPPY, None, "W"),
        "cheerful": (EyeState.HAPPY, None, "W"), "delighted": (EyeState.HAPPY, None, "W"),
        "friendly": (EyeState.HAPPY, None, "W"), "playful": (EyeState.EXCITED, None, "W"),
        "excited": (EyeState.EXCITED, None, "W"), "enthusiastic": (EyeState.EXCITED, None, "W"),
        "proud": (EyeState.PROUD, "up", "W"), "confident": (EyeState.PROUD, "up", None),
        "curious": (EyeState.CURIOUS, None, "PAN"), "interested": (EyeState.CURIOUS, None, "PAN"),
        "inquisitive": (EyeState.CURIOUS, None, "PAN"), "surprised": (EyeState.SURPRISED, None, "F"),
        "shocked": (EyeState.SURPRISED, None, "F"), "amazed": (EyeState.SURPRISED, None, "F"),
        "thinking": (EyeState.THINKING, None, "T"), "confused": (EyeState.THINKING, None, "T"),
        "uncertain": (EyeState.THINKING, None, "T"), "pondering": (EyeState.THINKING, None, "T"),
        "thoughtful": (EyeState.THINKING, None, "T"), "sad": (EyeState.SAD, "down", "B"),
        "unhappy": (EyeState.SAD, "down", "B"), "disappointed": (EyeState.SAD, "down", "B"),
        "sorry": (EyeState.SAD, "down", "B"), "apologetic": (EyeState.SAD, "down", "B"),
        "angry": (EyeState.ANGRY, None, "B"), "frustrated": (EyeState.ANGRY, None, "B"),
        "annoyed": (EyeState.ANGRY, None, "B"),
    }

    def __init__(self, config: Optional[Config] = None, settings: Optional[RuntimeSettings] = None):
        self.config = config or Config.from_env()
        self.settings = settings or RuntimeSettings()
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # ── State ─────────────────────────────────────────────────────────────
        self.running = False
        self.sleep_mode = False
        self.is_speaking = False
        self.active_user: Optional[str] = None
        self.last_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.current_objects: list = []
        self.current_detections: list = []
        self.last_faces: list = []
        self.persistent_objects: set = set()
        self._vision_state_lock = threading.Lock()
        self._camera_task_lock = threading.Lock()
        self._last_logged_face_present: Optional[bool] = None
        self._last_logged_recognized_name: Optional[str] = None
        self._last_logged_objects: tuple = ()
        self._thinking_token: Optional[str] = None
        self._camera_thread: Optional[threading.Thread] = None
        self._keyboard_thread: Optional[threading.Thread] = None
        self._notif_queue: list = []
        self._notif_lock = threading.Lock()
        self.local_vision_enabled = False
        self.detector = None
        self.recognizer = None
        self.object_detector = None
        self.display_enabled = self.settings.display_enabled
        self._cleaned_up = False
        self._stream_frame: Optional[bytes] = None
        self._stream_lock = threading.Lock()
        self._text_mode_active = False
        self._pc_stream_active = False
        self._pc_stream_thread: Optional[threading.Thread] = None
        self._response_time_stats_path = ROOT_DIR / "memory" / "response_time_stats.json"
        self._response_time_stats: dict = {}
        self._emergency_active = False
        self._last_voice_stats: dict = {}
        self._previous_behavior_frame: Optional[np.ndarray] = None
        self._last_motion_time = time.time()
        self._last_visual_check_time = 0.0
        self._visual_check_active = False
        self._pause_wake_listening = False
        self.follow_mode = False
        self._follow_last_command = "S"
        self._follow_last_command_time = 0.0
        self._follow_last_seen_time = 0.0
        self._eye_track_last_update = 0.0
        self._last_wake_text = ""
        self.servo = None
        self.servo_enabled = False
        self._servo_face_last_update = 0.0
        self._servo_face_tracking_enabled = False
        self._bbox_fall_started_at: Optional[float] = None
        self._bbox_fall_last_alert_time = 0.0
        self._blood_detect_started_at: Optional[float] = None
        self._blood_last_alert_time = 0.0
        self._ultrasonic_sensor = None
        self._ultrasonic_thread: Optional[threading.Thread] = None
        self._ultrasonic_blocked = False
        self._ultrasonic_distance_m: Optional[float] = None
        self._ultrasonic_last_announce_time = 0.0
        self._clap_thread: Optional[threading.Thread] = None
        self._clap_last_time = 0.0
        self._clap_cooldown = 3.0
        self._clap_enabled = os.getenv("BUDDY_ENABLE_CLAP", "1") == "1"
        self._listen_loop: Optional[asyncio.AbstractEventLoop] = None
        self._working_arecord_device: Optional[str] = None
        self._listen_initial_timeout: Optional[float] = None
        self._stt_endpoint_warned = False
        self._camera_tasks: dict = {
            "objects": _CameraTask("objects", "_run_object_detection_task", max(1, self.settings.object_interval_frames)),
            "faces": _CameraTask("faces", "_run_face_detection_task", max(1, self.settings.face_interval_frames)),
            "pose_behavior": _CameraTask("pose_behavior", "_run_pose_behavior_task", 3),
        }

        # ── Surveillance ──────────────────────────────────────────────────────
        self._surveillance_client: Optional[_SurveillanceClient] = None
        if self.settings.surveillance_enabled:
            self._surveillance_client = _SurveillanceClient(
                pc_ip=self.settings.pc_camera_ip,
                port=self.settings.surveillance_port,
                alert_callback=self._on_surveillance_event,
                cooldown=self.settings.surveillance_cooldown,
            )

        # ── Behavior pipeline ─────────────────────────────────────────────────
        self._behavior_pipeline = None
        if os.getenv("BUDDY_ENABLE_MEDIAPIPE_BEHAVIOR", "0") == "1":
            self._behavior_pipeline = BehaviorDetectionPipeline(
                config=PipelineConfig(resize_width=224, resize_height=224,
                                      process_every_n_frames=3, sequence_length=12,
                                      enable_visualization=False),
                alert_callback=self._on_behavior_alert,
            )

        # ── Audio / camera / hardware ─────────────────────────────────────────
        self.aplay_device, self.usb_card_index = self._find_output_audio_device()
        self.arecord_device = self._find_input_audio_device()
        self._init_camera()
        self.stability = StabilityTracker(self.config)
        self._init_local_vision()

        motor_port = self.settings.arduino_port or None
        self.motors = MotorController(port=motor_port, baud=self.settings.arduino_baud)
        self.motors.set_obstacle_callback(self._on_obstacle)
        self.motors.set_clear_callback(self._on_clear_path)

        self._init_servo()
        self._init_ultrasonic_sensor()
        self._load_response_time_stats()

        self.tts_voice = "en-IN-NeerjaNeural"
        self.tts_gain = max(0.5, float(os.getenv("BUDDY_TTS_GAIN", "2.0")))
        self.listen_beep_gain = min(1.0, max(0.1, float(os.getenv("BUDDY_LISTEN_BEEP_GAIN", "0.75"))))
        self.speech_enabled = True

        self.eyes: Optional[OledEyes] = None
        try:
            self.eyes = OledEyes()
        except Exception as exc:
            self.logger.warning("OLED eyes unavailable: %s", exc)

        atexit.register(self.cleanup)

    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.log_level, logging.INFO),
            format=self.config.log_format,
        )

    def run(self):
        self.running = True
        print("[Safety] Passive camera and emergency-phrase detection enabled.")
        print("[Input] Type commands or use voice.")
        self._start_phone_listener()
        self._start_stream_server(port=8090)
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        if self._surveillance_client:
            self._surveillance_client.start()
        threading.Thread(target=self._startup_greeting, daemon=True).start()
        threading.Thread(target=self._text_mode_loop, daemon=True).start()
        self._start_keyboard_listener()
        if self._clap_enabled:
            self._clap_thread = threading.Thread(target=self._clap_detection_loop, daemon=True)
            self._clap_thread.start()
            print("[Clap] Detection enabled")
        time.sleep(0.5)
        self._wake_loop()

    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self.running = False
        if self._surveillance_client:
            self._surveillance_client.stop()
        if self.eyes:
            try: self.eyes.stop()
            except Exception: pass
        try: self.motors.cleanup()
        except Exception: pass
        if self._ultrasonic_sensor:
            try: self._ultrasonic_sensor.close()
            except Exception: pass
            finally: self._ultrasonic_sensor = None
        if self.servo_enabled and self.servo:
            try: self.servo.cleanup()
            except Exception: pass
            finally: self.servo_enabled = False
        self._pc_stream_active = False
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
        if self.csi_process:
            try: self.csi_process.terminate()
            except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> int:
    buddy = None
    try:
        buddy = BuddyIntegratedPi()

        def _shutdown(sig, frame):
            if buddy: buddy.cleanup()
            raise SystemExit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        buddy.run()
        return 0
    except KeyboardInterrupt:
        if buddy: buddy.cleanup()
        return 0
    except Exception as exc:
        logging.error("Buddy startup failed: %s", exc)
        if buddy: buddy.cleanup()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
