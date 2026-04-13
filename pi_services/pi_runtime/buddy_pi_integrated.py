"""
Integrated Buddy runtime for Raspberry Pi.

This file combines:
- the conversation/registration behavior from buddy_windows_test.py
- the phone notification flow from buddy_phone_link.py
- Pi-friendly camera/audio/motor control

Run from pi_services:
    python pi_runtime/buddy_pi_integrated.py
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import edge_tts
import numpy as np
import requests
from scipy.signal import resample_poly

from core.config import Config
from core.stability_tracker import StabilityTracker
from hardware.motor_controller import MotorController
from memory.pi_memory import delete_person
from phone_link.core import process_notification

os.environ["LIBCAMERA_LOG_LEVELS"] = "*:ERROR"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"


@dataclass
class RuntimeSettings:
    stt_server_ip: str = os.getenv("BUDDY_STT_SERVER_IP", "192.168.0.105")
    stt_port: int = int(os.getenv("BUDDY_STT_PORT", "8765"))
    notification_port: int = int(os.getenv("BUDDY_NOTIFICATION_PORT", "8001"))
    arduino_port: str = os.getenv("BUDDY_ARDUINO_PORT", "/dev/ttyUSB0")
    arduino_baud: int = int(os.getenv("BUDDY_ARDUINO_BAUD", "115200"))
    use_servo: bool = False
    recognition_interval: float = 5.0
    object_interval_frames: int = 20
    face_interval_frames: int = 15
    display_enabled: bool = os.getenv("BUDDY_ENABLE_DISPLAY", "1") != "0"


class BuddyIntegratedPi:
    _WAKE_WORDS = ("buddy", "hey buddy", "hi buddy")
    _SLEEP_PHRASES = ("go to sleep", "sleep now", "take a nap", "good night")
    _STOP_WORDS = ("stop", "halt", "freeze", "don't move", "do not move")
    _DIR_MAP = {
        "forward": "F",
        "ahead": "F",
        "backward": "B",
        "back": "B",
        "left": "L",
        "right": "R",
    }
    _MOTOR_CONFIRMATIONS = {
        "F": "Moving forward.",
        "B": "Moving backward.",
        "L": "Turning left.",
        "R": "Turning right.",
        "S": "Stopped.",
    }
    _THINK_SOUNDS = (
        "Let me think.",
        "One moment.",
        "Thinking.",
    )

    def __init__(self, config: Optional[Config] = None, settings: Optional[RuntimeSettings] = None):
        self.config = config or Config.from_env()
        self.settings = settings or RuntimeSettings()
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        self.running = False
        self.sleep_mode = False
        self.is_speaking = False
        self.awaiting_name = False
        self.active_user: Optional[str] = None
        self.last_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.last_recognition_time = 0.0
        self.current_objects: list[str] = []
        self.current_detections = []
        self.last_faces = []
        self.persistent_objects: set[str] = set()
        self._thinking_token: Optional[str] = None
        self._camera_thread: Optional[threading.Thread] = None
        self._notif_queue: list[dict] = []
        self._notif_lock = threading.Lock()
        self.local_vision_enabled = False
        self.detector = None
        self.recognizer = None
        self.object_detector = None
        self.display_enabled = self.settings.display_enabled
        self._cleaned_up = False

        self.aplay_device, self.usb_card_index = self._find_output_audio_device()
        self.arecord_device = self._find_input_audio_device()
        self._listen_loop: Optional[asyncio.AbstractEventLoop] = None
        self._working_arecord_device: Optional[str] = None

        self._init_camera()
        self.stability = StabilityTracker(self.config)
        self._init_local_vision()
        self.motors = MotorController(
            port=self.settings.arduino_port,
            baud=self.settings.arduino_baud,
        )
        self.motors.set_obstacle_callback(self._on_obstacle)
        self.motors.set_clear_callback(self._on_clear_path)

        self.tts_voice = "en-IN-NeerjaNeural"
        self.speech_enabled = True

        atexit.register(self.cleanup)

    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.log_level, logging.INFO),
            format=self.config.log_format,
        )

    def _probe_local_vision_stack(self) -> bool:
        if os.getenv("BUDDY_DISABLE_LOCAL_VISION", "0") == "1":
            self.logger.warning("Local vision disabled by BUDDY_DISABLE_LOCAL_VISION=1")
            return False

        probe = "import onnxruntime\nprint('vision-ok')\n"
        try:
            result = subprocess.run(
                [sys.executable, "-c", probe],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
                text=True,
            )
            if result.returncode == 0 and "vision-ok" in result.stdout:
                return True
            self.logger.warning("Local vision probe failed: %s", (result.stderr or result.stdout).strip())
            return False
        except Exception as exc:
            self.logger.warning("Local vision probe exception: %s", exc)
            return False

    def _init_local_vision(self):
        if not self._probe_local_vision_stack():
            self.local_vision_enabled = False
            self.logger.warning("Buddy will start without local face/object inference on this Pi.")
            return

        from vision.opencv_face_detector import FaceDetector
        from vision.local_face_recognizer import FaceRecognizer
        from vision.objrecog.obj import ObjectDetector

        self.detector = FaceDetector(self.config.cascade_path, self.config)
        self.recognizer = FaceRecognizer(self.config.model_path, self.config)
        self.object_detector = ObjectDetector(confidence_threshold=0.5)
        self.local_vision_enabled = True
        self.logger.info("Local vision initialized successfully")

    def _init_camera(self):
        self.csi_frame_path = "/tmp/csi_frame.jpg"
        self.csi_process = None
        self.use_csi = False
        self.cap = None

        helper_path = ROOT_DIR / "hardware" / "csi_camera_helper.py"
        if shutil.which("rpicam-hello") and helper_path.exists():
            try:
                self.csi_process = subprocess.Popen(
                    ["/usr/bin/python3", str(helper_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                for _ in range(50):
                    if os.path.exists(self.csi_frame_path):
                        self.use_csi = True
                        self.logger.info("CSI camera ready")
                        return
                    time.sleep(0.1)
            except Exception as exc:
                self.logger.warning("CSI helper failed: %s", exc)

        for idx in (self.config.camera_index, 0, 1, 2):
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.camera_buffer_size)
            cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            ok, _ = cap.read()
            if ok:
                self.cap = cap
                self.logger.info("USB camera opened at index %s", idx)
                return
            cap.release()

        raise RuntimeError("No camera available for Buddy")

    def _read_frame(self):
        if self.use_csi and os.path.exists(self.csi_frame_path):
            frame = cv2.imread(self.csi_frame_path)
            return (True, frame) if frame is not None else (False, None)
        if self.cap is not None:
            return self.cap.read()
        return False, None

    def _scan_alsa_cards(self, tool: str) -> list[tuple[str, str, str]]:
        results = []
        try:
            result = subprocess.run([tool, "-l"], capture_output=True, text=True, timeout=5)
            for line in result.stdout.splitlines():
                if not line.startswith("card"):
                    continue
                try:
                    card_num = line.split(":")[0].strip().split()[-1]
                    dev_num = line.split("device")[-1].strip().split(":")[0].strip()
                    card_name = line.split(":")[1].split("[")[0].strip().replace(" ", "_")
                    results.append((card_num, card_name, dev_num))
                except Exception:
                    continue
        except Exception:
            pass
        return results

    def _find_output_audio_device(self) -> tuple[str, str]:
        cards = self._scan_alsa_cards("aplay")
        for card_num, card_name, dev_num in cards:
            if "usb" in card_name.lower() or "bcm2835" in card_name.lower() or "headphone" in card_name.lower():
                return f"plughw:{card_num},{dev_num}", card_num
        return "default", "0"

    def _find_input_audio_device(self) -> str:
        cards = self._scan_alsa_cards("arecord")
        for card_num, card_name, dev_num in cards:
            if any(token in card_name.lower() for token in ("usb", "seeed", "snd", "audio", "mic", "i2s")):
                return f"plughw:{card_num},{dev_num}"
        return "default"

    def _generate_beep_wav(self) -> str:
        sample_rate = 22050
        duration = 0.14
        tones = [(740, duration), (920, duration)]
        frames = []
        for freq, dur in tones:
            t = np.linspace(0, dur, int(sample_rate * dur), endpoint=False)
            tone = 0.3 * np.sin(2 * np.pi * freq * t)
            frames.append((tone * 32767).astype(np.int16).tobytes())

        wav_path = tempfile.mktemp(suffix=".wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
        return wav_path

    def _play_listen_beep(self):
        try:
            wav_path = self._generate_beep_wav()
            for device in (self.aplay_device, "default"):
                result = subprocess.run(
                    ["aplay", "-D", device, "-q", wav_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=4,
                )
                if result.returncode == 0:
                    break
        except Exception as exc:
            self.logger.warning("Listen beep failed: %s", exc)

    def speak(self, text: str):
        if not text.strip():
            return
        self.is_speaking = True

        def _run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._tts(text))
                loop.close()
            except Exception as exc:
                self.logger.warning("TTS failed: %s", exc)
            finally:
                self.is_speaking = False

        threading.Thread(target=_run, daemon=True).start()

    async def _tts(self, text: str):
        communicate = edge_tts.Communicate(text, self.tts_voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        if not audio_data:
            return

        mp3_path = tempfile.mktemp(suffix=".mp3")
        with open(mp3_path, "wb") as handle:
            handle.write(audio_data)
        try:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", mp3_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
        except Exception:
            subprocess.run(
                ["aplay", mp3_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
        finally:
            try:
                os.remove(mp3_path)
            except OSError:
                pass

    def _wait_for_tts(self, timeout: float = 20.0):
        start = time.time()
        while self.is_speaking and (time.time() - start) < timeout:
            time.sleep(0.05)
        time.sleep(0.2)

    def _record_audio_vad(self) -> np.ndarray:
        mic_rate = 48000
        target_chunk_secs = 0.1
        speech_thresh = 0.008
        silence_thresh = 0.003
        silence_after = 0.4
        min_speech = 0.4
        max_duration = 8.0

        candidates = [self.arecord_device, "default"]
        if self._working_arecord_device:
            candidates.insert(0, self._working_arecord_device)

        working = None
        for device in candidates:
            try:
                probe = tempfile.mktemp(suffix=".wav")
                result = subprocess.run(
                    ["arecord", "-D", device, "-f", "S16_LE", "-r", str(mic_rate), "-c", "1", "-d", "1", probe],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=4,
                )
                try:
                    os.remove(probe)
                except OSError:
                    pass
                if result.returncode == 0:
                    working = device
                    self._working_arecord_device = device
                    break
            except Exception:
                continue

        if not working:
            return np.array([], dtype=np.float32)

        bytes_per_chunk = int(mic_rate * target_chunk_secs) * 2
        proc = subprocess.Popen(
            ["arecord", "-D", working, "-f", "S16_LE", "-r", str(mic_rate), "-c", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        speech_started = False
        silence_duration = 0.0
        speech_duration = 0.0
        total_duration = 0.0
        chunks: list[np.ndarray] = []

        try:
            while total_duration < max_duration:
                raw = proc.stdout.read(bytes_per_chunk)
                if not raw or len(raw) < bytes_per_chunk:
                    break
                chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                total_duration += target_chunk_secs

                if rms >= speech_thresh:
                    speech_started = True
                    silence_duration = 0.0
                    speech_duration += target_chunk_secs
                    chunks.append(chunk)
                elif speech_started:
                    silence_duration += target_chunk_secs
                    chunks.append(chunk)
                    if silence_duration >= silence_after:
                        break
        finally:
            proc.kill()
            proc.wait()

        if not speech_started or speech_duration < min_speech:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    async def _ws_listen_once(self) -> str:
        import websockets

        audio = await asyncio.get_event_loop().run_in_executor(None, self._record_audio_vad)
        if audio.size == 0:
            return ""

        audio_16k = resample_poly(audio, 16000, 48000).astype(np.float32)
        uri = f"ws://{self.settings.stt_server_ip}:{self.settings.stt_port}"
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(audio_16k.tobytes())
                result = await websocket.recv()
                return result.strip() if result else ""
        except Exception as exc:
            self.logger.warning("STT websocket failed: %s", exc)
            return ""

    def listen_for_speech(self) -> str:
        try:
            if self._listen_loop is None or self._listen_loop.is_closed():
                self._listen_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._listen_loop)
            text = self._listen_loop.run_until_complete(self._ws_listen_once())
            return text if len(text) > 1 else ""
        except Exception as exc:
            self.logger.warning("listen_for_speech failed: %s", exc)
            self._listen_loop = None
            return ""

    def _call_brain(self, user_input: str, recognized_user: Optional[str] = None) -> dict:
        objects = list(self.current_objects or self.persistent_objects)
        try:
            response = requests.post(
                f"{self.config.llm_service_url}/chat",
                json={
                    "user_input": user_input,
                    "recognized_user": recognized_user,
                    "objects_visible": objects,
                },
                timeout=30,
            )
            if response.status_code == 200:
                return response.json()
        except Exception as exc:
            self.logger.warning("Brain call failed: %s", exc)
        return {
            "reply": "Sorry, I'm having trouble thinking right now.",
            "intent": "conversation",
            "emotion": "apologetic",
            "raw_response": "",
        }

    def _play_thinking_sound(self):
        token = f"{time.time()}-{random.random()}"
        self._thinking_token = token

        def _run():
            time.sleep(1.2)
            if self._thinking_token == token:
                self.speak(random.choice(self._THINK_SOUNDS))

        threading.Thread(target=_run, daemon=True).start()

    def _extract_name(self, text: str) -> str:
        not_names = {
            "here", "there", "not", "just", "also", "very", "really", "sorry",
            "good", "fine", "okay", "ok", "yes", "no", "sure", "well", "now",
            "still", "already", "going", "trying", "doing", "being", "having",
            "little", "bit", "back", "home", "ready", "happy", "busy", "tired",
        }
        lowered = text.lower().strip()
        for phrase in ("my name is", "i am", "i'm", "call me", "name is"):
            if phrase not in lowered:
                continue
            after = lowered.split(phrase, 1)[1].strip()
            for word in after.split():
                clean = "".join(ch for ch in word if ch.isalpha())
                if len(clean) > 1 and clean not in not_names:
                    return clean.title()
        return ""

    def _register_multi_angle(self, name: str):
        if not self.local_vision_enabled or self.detector is None or self.recognizer is None:
            self.logger.warning("Skipping face registration because local vision is unavailable")
            return

        poses = [
            ("front", "Great. Turn your face slowly to the left and hold still."),
            ("left", "Perfect. Now turn to the right and hold still."),
            ("right", "Good. Tilt your face slightly upward."),
            ("up", "Nice. Now tilt your face slightly downward."),
            ("down", None),
        ]

        for angle, next_instruction in poses:
            time.sleep(3.0)
            captured = False
            for _ in range(25):
                if self.last_frame is None:
                    time.sleep(0.1)
                    continue
                faces = self.detector.detect(self.last_frame)
                if not faces:
                    time.sleep(0.1)
                    continue
                x, y, w, h = self.detector.get_largest_face(faces)
                if w <= 80 or h <= 80:
                    time.sleep(0.1)
                    continue
                face_roi = self.last_frame[y:y + h, x:x + w]
                if self.recognizer.add_face(name, face_roi, angle):
                    captured = True
                    break
                time.sleep(0.1)

            if not captured:
                self.logger.warning("Could not capture %s angle for %s", angle, name)
            if next_instruction:
                self.speak(next_instruction)
                self._wait_for_tts()

    def _process_registration_request(self, text: str) -> bool:
        if not self.local_vision_enabled:
            return False

        name = self._extract_name(text)
        if not name:
            return False

        self.active_user = name
        self.awaiting_name = False
        self.speak(
            f"Nice to meet you {name}. I will scan your face from different angles. Please look straight at me and hold still."
        )
        self._wait_for_tts()
        self._register_multi_angle(name)
        reply = self._call_brain(
            f"You just registered {name}'s face from multiple angles. Greet them warmly.",
            recognized_user=name,
        )
        self._display_response(reply)
        return True

    def _is_stop_command(self, text: str) -> bool:
        return any(word in text for word in self._STOP_WORDS)

    def _parse_movement(self, text: str):
        cmd = None
        for word, code in self._DIR_MAP.items():
            if word in text:
                cmd = code
                break
        if cmd is None:
            return None

        duration = None
        match = re.search(r"for\s+(\d+(?:\.\d+)?)\s*(second|sec|minute|min)", text)
        if match:
            value = float(match.group(1))
            duration = value * 60 if match.group(2).startswith("min") else value
        return cmd, duration

    def _process_input(self, text: str):
        lowered = text.lower()

        if self._is_stop_command(lowered):
            self.motors.stop()
            self.speak("Stopped.")
            return

        movement = self._parse_movement(lowered)
        if movement is not None:
            cmd, duration = movement
            self.motors.move(cmd, duration)
            self.speak(self._MOTOR_CONFIRMATIONS.get(cmd, "On it."))
            return

        if any(phrase in lowered for phrase in self._SLEEP_PHRASES):
            self.speak("Going to sleep. Say hey buddy to wake me up.")
            self._wait_for_tts()
            self.sleep_mode = True
            return

        if self._process_registration_request(text):
            return

        self._play_thinking_sound()
        response = self._call_brain(text, self.active_user)
        self._thinking_token = None
        self._display_response(response)

    def _display_response(self, response: dict):
        if not response:
            return
        reply = response.get("reply", "")
        intent = response.get("intent", "conversation")
        if intent == "stop":
            self.motors.stop()
        self.speak(reply)

    def _process_frame(self, frame: np.ndarray):
        if not self.local_vision_enabled or self.detector is None or self.recognizer is None or self.object_detector is None:
            return frame.copy()

        if self.frame_count % self.settings.object_interval_frames == 0:
            self.current_detections = self.object_detector.detect(frame)
            self.current_objects = [det["name"] for det in self.current_detections if det["name"] != "person"]
            self.persistent_objects.update(self.current_objects)

        if self.frame_count % self.settings.face_interval_frames == 0:
            self.last_faces = self.detector.detect(frame)
            if self.last_faces:
                largest = self.detector.get_largest_face(self.last_faces)
                is_stable = self.stability.update(largest)
                x, y, w, h = largest
                if is_stable and w > 80 and h > 80:
                    now = time.time()
                    if now - self.last_recognition_time > self.settings.recognition_interval:
                        face_roi = frame[y:y + h, x:x + w]
                        name, confidence = self.recognizer.recognize(face_roi)
                        self.last_recognition_time = now
                        if name != "Unknown" and confidence > 0.45:
                            if self.active_user != name:
                                self.active_user = name
                                reply = self._call_brain(
                                    f"You just recognized {name}. Greet them warmly by name.",
                                    recognized_user=name,
                                )
                                self._display_response(reply)
                        elif not self.awaiting_name and not self.is_speaking:
                            self.awaiting_name = True
                            reply = self._call_brain(
                                "You see someone but don't recognize them. Say hi and ask for their name.",
                                recognized_user=None,
                            )
                            self._display_response(reply)
            else:
                self.stability.reset()

        return self._draw_visualization(frame.copy())

    def _draw_visualization(self, frame: np.ndarray) -> np.ndarray:
        if not self.local_vision_enabled:
            status = "Local vision disabled on this Pi"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return frame

        if self.current_detections:
            frame = self.object_detector.draw_detections(frame, self.current_detections)

        for (x, y, w, h) in self.last_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if self.active_user:
                cv2.putText(
                    frame,
                    self.active_user,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        status = f"Chatting with: {self.active_user}" if self.active_user else "Looking for people..."
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame

    def _start_phone_listener(self):
        buddy = self

        class _ThreadedServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path not in ("/", "/notify"):
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers.get("Content-Length", "0"))
                payload = self.rfile.read(length)
                try:
                    data = json.loads(payload or b"{}")
                    result = process_notification(
                        data.get("app", "Unknown"),
                        data.get("title", ""),
                        data.get("message", ""),
                    )
                    if result.get("status") == "received":
                        threading.Thread(target=buddy._on_phone_notification, args=(result,), daemon=True).start()
                    body = json.dumps(result).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                except Exception:
                    self.send_response(400)
                    self.end_headers()

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"message":"Buddy phone link active"}')

            def log_message(self, *args):
                return

        def _run():
            server = _ThreadedServer(("0.0.0.0", self.settings.notification_port), _Handler)
            server.serve_forever()

        threading.Thread(target=_run, daemon=True).start()

    def _on_phone_notification(self, notification: dict):
        if notification.get("decision") == "ignore" or self.sleep_mode:
            return
        with self._notif_lock:
            self._notif_queue.append(notification)
        if not self.is_speaking:
            threading.Thread(target=self._flush_notifications, daemon=True).start()

    def _flush_notifications(self):
        while True:
            with self._notif_lock:
                if not self._notif_queue:
                    return
                notification = self._notif_queue.pop(0)
            self._wait_for_tts()
            prompt = (
                "You received a phone notification. Relay it casually in one short sentence. "
                f"App: {notification.get('app', '')}. "
                f"From: {notification.get('sender', '')}. "
                f"Message: {notification.get('message', '')}."
            )
            response = self._call_brain(prompt, recognized_user=self.active_user)
            self._display_response(response)
            self._wait_for_tts()

    def _on_obstacle(self):
        if not self.is_speaking:
            self.speak("Obstacle detected. I have stopped.")

    def _on_clear_path(self):
        if not self.is_speaking:
            self.speak("Path is clear.")

    def _camera_loop(self):
        while self.running:
            if self.sleep_mode:
                time.sleep(0.1)
                continue

            ok, frame = self._read_frame()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            self.last_frame = frame.copy()
            self.frame_count += 1
            processed = self._process_frame(frame)
            if self.display_enabled:
                try:
                    cv2.imshow("Buddy Vision", processed)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False
                        break
                except cv2.error as exc:
                    self.display_enabled = False
                    self.logger.warning("OpenCV display disabled: %s", exc)
            time.sleep(0.02)

    def _wake_loop(self):
        self.speak("Hey. I am Buddy. Call me anytime.")
        self._wait_for_tts()

        while self.running:
            if self.sleep_mode:
                text = self.listen_for_speech()
                if text and any(word in text.lower() for word in self._WAKE_WORDS):
                    self.sleep_mode = False
                    self.speak("I am awake. What can I do for you?")
                    self._wait_for_tts()
                continue

            text = self.listen_for_speech()
            if not text:
                continue

            lowered = text.lower()
            if not any(word in lowered for word in self._WAKE_WORDS):
                continue

            self._play_listen_beep()
            user_text = self.listen_for_speech()
            if user_text:
                self._process_input(user_text)
                self._wait_for_tts()

            with self._notif_lock:
                pending_notifications = bool(self._notif_queue)
            if pending_notifications:
                threading.Thread(target=self._flush_notifications, daemon=True).start()

    def run(self):
        self.running = True
        self._start_phone_listener()
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        self._wake_loop()

    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self.running = False
        try:
            self.motors.cleanup()
        except Exception:
            pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        if self.csi_process is not None:
            try:
                self.csi_process.terminate()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main() -> int:
    buddy = None
    try:
        buddy = BuddyIntegratedPi()

        def _shutdown(sig, frame):
            if buddy is not None:
                buddy.cleanup()
            raise SystemExit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        buddy.run()
        return 0
    except KeyboardInterrupt:
        if buddy is not None:
            buddy.cleanup()
        return 0
    except Exception as exc:
        logging.error("Integrated Buddy startup failed: %s", exc)
        if buddy is not None:
            buddy.cleanup()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
