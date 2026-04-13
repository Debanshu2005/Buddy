"""
Buddy Pi - Hardware Only Service
Face recognition, voice I/O, camera, objects - NO BRAIN
Optimized for Raspberry Pi 4B with Python 3.13
Uses Whisper STT for speech recognition
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

# Hardware modules only
from core.config import Config
from core.states import BuddyState, StateManager
from vision.face_detector import FaceDetector
from vision.face_recognizer import FaceRecognizer
from core.stability_tracker import StabilityTracker
from vision.objrecog.obj import ObjectDetector
from hardware.servo_controller import ServoController
from hardware.motor_controller import MotorController

# Speech imports - Whisper STT
try:
    from audio.speech_to_text import listen  # 🔑 Whisper STT
except ImportError as e:
    print(f"❌ Failed to import STT: {e}")
    listen = None

# TTS imports
import edge_tts
import asyncio
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
                    return f"plughw:{card_idx},0", card_idx
    except Exception as e:
        print(f"⚠️ USB audio detection error: {e}")

    print("⚠️ USB audio device not found via aplay, defaulting to plughw:1,0")
    return "plughw:1,0", "1"


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
        self.motors = MotorController()

        # Face tracking state
        self.face_lost_count = 0
        self.max_face_lost = 10

        # Initialize speech (Whisper STT + Edge TTS)
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
        """Initialize speech recognition (Whisper STT) and TTS (Edge TTS) for USB mic"""
        try:
            if listen is None:
                print("❌ Whisper STT not available - speech disabled")
                self.speech_enabled = False
                return

            # Initialize pygame mixer using the USB audio card
            # The 'devicename' parameter isn't available in all pygame versions,
            # so we target the USB card via SDL env var instead.
            os.environ["SDL_AUDIODRIVER"] = "alsa"
            os.environ["AUDIODEV"] = f"plughw:{self.usb_card_index},0"

            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.tts_voice = "en-IN-NeerjaNeural"

            self.speech_enabled = True
            print(f"✅ Speech system ready (Whisper STT + Edge TTS via USB mic, card {self.usb_card_index})")

        except Exception as e:
            print(f"⚠️ Speech initialization failed: {e}")
            self.speech_enabled = False

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

                    if name != "Unknown" and confidence > self.recognition_threshold:
                        if self.active_user != name:
                            self.active_user = name

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
        """Generate speech using Edge TTS and play via USB audio device"""
        try:
            communicate = edge_tts.Communicate(text, self.tts_voice)

            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if audio_data:
                temp_audio = "/tmp/tts_output.wav"
                with open(temp_audio, "wb") as f:
                    f.write(audio_data)

                # Play via USB audio device (auto-detected at startup)
                os.system(f"aplay -D {self.aplay_device} {temp_audio}")

        except Exception as e:
            print(f"Edge TTS Error: {e}")

    def listen_for_speech(self, timeout=6.0):
        """Listen for speech using Whisper STT via USB microphone"""
        if not self.speech_enabled or listen is None:
            print("⚠️ Speech system disabled")
            return ""

        try:
            print("🎤 Listening (USB mic)...")

            # Whisper STT picks up the default ALSA device, which is set to the
            # USB card via the AUDIODEV env var configured in _init_speech()
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

            if intent in ["move_forward", "move_backward", "move_left", "move_right", "stop"]:
                self._execute_movement(intent)

        self.speak(reply)

        words = len(reply.split())
        chars = len(reply)
        reading_time = max(2.0, (words * 0.3) + (chars * 0.05))
        reading_time = min(reading_time, 8.0)

        threading.Timer(reading_time, self._delayed_listening).start()

    def _execute_movement(self, intent: str):
        """Execute movement commands based on intent"""
        movement_map = {
            "move_forward": lambda: self.motors.move_forward(1.0),
            "move_backward": lambda: self.motors.move_backward(1.0),
            "move_left": lambda: self.motors.turn_left(0.5),
            "move_right": lambda: self.motors.turn_right(0.5),
            "stop": lambda: self.motors.stop()
        }

        if intent in movement_map:
            print(f"🤖 Executing: {intent}")
            movement_map[intent]()

    def _play_thinking_sound(self):
        """Play thinking sound while processing"""
        thinking_sounds = ["Hmm", "Let me think", "Umm", "Well"]
        import random
        sound = random.choice(thinking_sounds)
        self.speak(sound)
        time.sleep(0.3)

    def _delayed_listening(self):
        """Start listening after a delay"""
        if not self.running or self.sleep_mode:
            return

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

    def _process_input(self, user_text):
        """Process user input"""
        try:
            if any(phrase in user_text.lower() for phrase in ['my name is', 'i am', "i'm"]):
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

            response = self._call_brain_service(user_text, self.active_user)
            self._display_response(response)
        except Exception as e:
            print(f"Error processing input: {e}")

    def _enter_sleep_mode(self):
        """Trigger sleep mode - ends conversation loop"""
        print("\n😴 Entering sleep mode...")
        self.sleep_mode = True

    def _wake_up_and_restart(self):
        """Wake up and restart conversation loop"""
        print("😊 Waking up!")
        self.sleep_mode = False
        self.failed_listen_attempts = 0

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

                text = listen() if listen else ""

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
