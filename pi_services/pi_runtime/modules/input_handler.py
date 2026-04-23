"""Input handler mixin — command processing, movement, brain calls, face registration."""
from __future__ import annotations
import os, random, re, threading, time
from typing import Optional
import numpy as np
import requests
from hardware.oled_eyes import EyeState

class InputHandlerMixin:
    def _handle_servo_command(self, lowered: str) -> bool:
        if "look up" in lowered:
            if not self.servo_enabled or not self.servo:
                self.speak("I cannot move my head right now because the servo is not available.")
                return True
            threading.Thread(target=self.servo.look_up, daemon=True).start()
            self.speak("Looking up.")
            return True

        if "look down" in lowered:
            if not self.servo_enabled or not self.servo:
                self.speak("I cannot move my head right now because the servo is not available.")
                return True
            threading.Thread(target=self.servo.look_down, daemon=True).start()
            self.speak("Looking down.")
            return True

        if "look center" in lowered or "look straight" in lowered:
            if not self.servo_enabled or not self.servo:
                self.speak("I cannot move my head right now because the servo is not available.")
                return True
            threading.Thread(target=self.servo.look_center, daemon=True).start()
            self.speak("Looking straight ahead.")
            return True

        return False


    def _call_brain(self, user_input: str, recognized_user: Optional[str] = None) -> dict:
        user_lower = user_input.lower()
        current_objects, persistent_objects = self._get_objects_snapshot()

        if any(p in user_lower for p in ("what is", "what do you see", "in my hand", "holding")):
            # use already-detected objects from camera loop — no extra detection call
            if not current_objects and self.local_vision_enabled and self.object_detector:
                ok, frame = self._read_frame()
                if ok and frame is not None:
                    fresh = self.object_detector.detect(frame)
                    if fresh:
                        self._set_object_results(fresh)
                        current_objects, persistent_objects = self._get_objects_snapshot()

        objects = current_objects or list(persistent_objects)
        print(f"[LLM] Sending: user='{recognized_user or self.active_user or 'unknown'}' text='{user_input}' objects={objects}")
        try:
            response = requests.post(
                f"{self.config.llm_service_url}/chat",
                json={
                    "user_input": user_input,
                    "recognized_user": recognized_user,
                    "objects_visible": objects,
                },
                timeout=90,
            )
            if response.status_code == 200:
                payload = response.json()
                print(f"[LLM] Reply received: intent={payload.get('intent', 'conversation')} emotion={payload.get('emotion', '')}")
                return payload
            self.logger.warning("Brain returned status %s", response.status_code)
            print(f"[LLM] Request failed with status {response.status_code}")
        except Exception as exc:
            self.logger.warning("Brain call failed: %s", exc)
            print(f"[LLM] Error: {exc}")
        return {
            "reply": "Sorry, I'm having trouble thinking right now.",
            "intent": "conversation",
            "emotion": "apologetic",
            "raw_response": "",
        }


    def _play_thinking_sound(self):
        token = f"{time.time()}-{random.random()}"
        self._thinking_token = token
        self._eye(EyeState.THINKING)
        self.motors.emotion_move("thinking")

        def _run():
            time.sleep(0.2)
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


    def _is_identity_check(self, text: str) -> bool:
        lowered = text.lower()
        return any(phrase in lowered for phrase in self._IDENTITY_TRIGGERS)


    def _process_input(self, text: str):
        lowered = text.lower()

        # 0. Emergency
        if self._is_emergency_phrase(lowered):
            self._trigger_emergency_response(text)
            return

        # 1. Stop
        if self._is_stop_command(lowered):
            self.follow_mode = False
            self.motors.stop()
            self.speak("Stopped.")
            return

        # 2. Follow mode
        if any(p in lowered for p in ("follow me", "come with me", "track me")):
            self._start_follow_mode()
            return
        if any(p in lowered for p in ("stop following", "don't follow", "do not follow")):
            self._stop_follow_mode()
            return

        # 2. Servo
        if self._handle_servo_command(lowered):
            return

        # 3. Move
        movement = self._parse_movement(lowered)
        if movement is not None:
            self.follow_mode = False
            cmd, duration = movement
            if not self._movement_allowed(cmd):
                return
            self.motors.move(cmd, duration)
            self.speak(self._MOTOR_CONFIRMATIONS.get(cmd, "On it."))
            # if no duration — listen for stop in background via STT
            if duration is None and cmd != "S":
                threading.Thread(target=self._listen_for_stop_command, daemon=True).start()
            return

        # 4. Sleep
        if any(phrase in lowered for phrase in self._SLEEP_PHRASES):
            self.sleep_mode = True
            return

        # 5. Register face
        if any(p in lowered for p in ("register my face", "register face", "add my face", "save my face")):
            self.speak("Sure! What's your name?")
            self._wait_for_tts()
            time.sleep(0.1)
            print("[Registration] Listening for name...")
            self._play_listen_beep()
            name_text = self.listen_for_speech()
            if name_text:
                name = self._extract_name(name_text)
                if not name:
                    words = [w.strip(".,!?") for w in name_text.split() if w.isalpha() and len(w) > 1]
                    name = words[0].title() if words else ""
            else:
                name = ""
            if not name:
                print("[Registration] Could not get name")
                self.speak("Sorry, I didn't catch your name. Please try again.")
                return
            print(f"[Registration] Name heard: {name}")

            self.speak(f"Got it {name}! Now please set a password. Say any word or phrase you will remember.")
            self._wait_for_tts()
            time.sleep(0.1)
            print("[Registration] Listening for password...")
            self._play_listen_beep()
            password_text = self.listen_for_speech()
            if not password_text:
                print("[Registration] Could not get password")
                self.speak("Sorry, I didn't catch your password. Please try again.")
                return
            password = password_text.lower().strip()
            print(f"[Registration] Password heard: '{password}'")
            from memory.pi_memory import save_password
            save_password(name, password)

            self.active_user = name
            self.speak(f"Password set! Now I'll scan your face from different angles. Please look straight at me and hold still.")
            self._wait_for_tts()
            threading.Thread(target=self._do_scan_then_save, args=(name,), daemon=True).start()
            return

        # 6. Identity check — single scan
        if self._is_identity_check(lowered):
            self._check_and_greet_face()
            return

        # 7. Brain
        self._play_thinking_sound()
        response = self._call_brain(text, self.active_user)
        self._thinking_token = None
        self._display_response(response)


    def _check_and_greet_face(self):
        """Scan face directly against all stored embeddings. Fall back to password if no match."""
        from memory.pi_memory import find_name_by_password, get_all_names_with_passwords

        if not self.local_vision_enabled or self.recognizer is None or self.detector is None:
            self.speak("Face recognition is not available right now.")
            self._wait_for_tts()
            return

        # Reload persisted faces first so recognition still works after a restart.
        try:
            self.recognizer.load_known_faces()
        except Exception as exc:
            self.logger.warning("Could not refresh known faces before identity check: %s", exc)

        if not self.recognizer.known_faces:
            self.speak("No one is registered yet. Say register my face to get started.")
            self._wait_for_tts()
            return

        # step 1: scan face directly
        self.speak("Let me scan your face. Please look at the camera.")
        self._wait_for_tts()
        print("[Face check] Scanning face...")

        scores: dict[str, float] = {}
        for attempt in range(25):
            ok, frame = self._read_frame()
            frame = frame if ok and frame is not None else self.last_frame
            if frame is None:
                time.sleep(0.1)
                continue
            faces = self.detector.detect(frame)
            if not faces:
                time.sleep(0.1)
                continue
            x, y, w, h = self.detector.get_largest_face(faces)
            if w < 60 or h < 60:
                time.sleep(0.1)
                continue
            emb = self.recognizer.get_embedding(frame[y:y + h, x:x + w])
            if emb is None:
                time.sleep(0.1)
                continue
            for name, stored_embs in self.recognizer.known_faces.items():
                best = max((float(np.dot(emb, s)) for s in stored_embs), default=0.0)
                scores[name] = max(scores.get(name, 0.0), best)
            print(f"[Face check] Scores: { {n: f'{s:.3f}' for n, s in scores.items()} }")
            time.sleep(0.15)
            if attempt >= 4 and scores:
                break

        if scores:
            best_name = max(scores, key=lambda n: scores[n])
            best_score = scores[best_name]
            print(f"[Face check] Best match: {best_name} ({best_score:.3f})")
            if best_score >= 0.32:
                self.active_user = best_name
                self.speak(f"Yes, I know you! Hi {best_name}!")
                self._wait_for_tts()
                return

        print("[Face check] No face match — trying password fallback")

        # step 2: password fallback
        registered = get_all_names_with_passwords()
        if not registered:
            self.speak("I couldn't recognize your face and no passwords are registered.")
            self._wait_for_tts()
            return

        self.speak("I couldn't recognize your face. Please say your password.")
        self._wait_for_tts()
        time.sleep(0.1)
        self._play_listen_beep()
        self._eye(EyeState.LISTENING)
        password_text = self.listen_for_speech()
        self._eye(EyeState.IDLE)
        if not password_text:
            self.speak("I didn't catch that. I don't know you.")
            self._wait_for_tts()
            return

        password = password_text.lower().strip()
        print(f"[Password check] Heard: '{password}'")
        found_name = find_name_by_password(password)
        if found_name:
            self.active_user = found_name
            self.speak(f"Hi {found_name}! Good to see you.")
        else:
            self.speak("Sorry, I don't know you.")
        self._wait_for_tts()


    def _do_scan_then_save(self, name: str):
        """
        Capture 4-5 face frames, save embeddings via recognizer + save photos locally.
        Photos saved to memory/face_photos/{name}/frame_N.jpg
        """
        import cv2 as _cv2
        from pathlib import Path as _Path
        photos_dir = _Path(__file__).resolve().parent.parent / "memory" / "face_photos" / name.lower()
        photos_dir.mkdir(parents=True, exist_ok=True)

        target_frames = 5
        captured = 0
        attempt = 0
        max_attempts = 60

        print(f"[Registration] Capturing {target_frames} face frames for {name}...")

        while captured < target_frames and attempt < max_attempts:
            attempt += 1
            ok, frame = self._read_frame()
            frame = frame if ok and frame is not None else self.last_frame
            if frame is None:
                time.sleep(0.15)
                continue

            if self.detector is None:
                time.sleep(0.15)
                continue

            faces = self.detector.detect(frame)
            if not faces:
                if attempt % 10 == 0:
                    print(f"[Registration] No face detected, retrying... ({attempt}/{max_attempts})")
                time.sleep(0.15)
                continue

            x, y, w, h = self.detector.get_largest_face(faces)
            if w < 60 or h < 60:
                time.sleep(0.15)
                continue

            face_roi = frame[y:y + h, x:x + w]

            # save embedding via recognizer
            if self.recognizer is not None:
                self.recognizer.add_face(name, face_roi, f"frame_{captured}")

            # save photo locally
            photo_path = photos_dir / f"frame_{captured}.jpg"
            _cv2.imwrite(str(photo_path), face_roi)
            captured += 1
            print(f"[Registration] ✅ Frame {captured}/{target_frames} captured")

            # small pause between captures so frames are different
            time.sleep(0.4)

        if captured >= 3:
            print(f"[Registration] ✅ {name} registered with {captured} frames")
            self.speak(f"Done! I've saved {captured} photos of your face {name}. I'll recognize you next time.")
        else:
            print(f"[Registration] ⚠️ Only {captured} frames captured for {name}")
            self.speak(f"I only captured {captured} photos {name}, but your password is saved. I'll try my best to recognize you.")
        self._wait_for_tts()


    def _is_stop_command(self, text: str) -> bool:
        return any(word in text for word in self._STOP_WORDS)


    def _parse_movement(self, text: str):
        cmd = None
        # check longer phrases first so "spin left" matches before "left"
        for word, code in sorted(self._DIR_MAP.items(), key=lambda x: -len(x[0])):
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


    def _apply_emotion(self, emotion: str):
        """Set OLED eye and motor animation from emotion string."""
        entry = self._EMOTION_MAP.get(emotion.lower().strip())
        if not entry:
            return
        eye_state, servo_action, motor_cmd = entry
        print(f"[EMOTION: {emotion}] eye={eye_state.value} servo={servo_action} motor={motor_cmd}")

        self._eye(eye_state)

        if motor_cmd and self.motors.is_connected() and self._movement_allowed(motor_cmd, announce=False):
            threading.Timer(0.3, lambda e=motor_cmd: self.motors.emotion_move(e)).start()


    def _display_response(self, response: dict):
        if not response:
            return
        reply  = response.get("reply", "")
        intent = response.get("intent", "conversation")
        emotion = response.get("emotion", "")
        print(f"\nBuddy: {reply}")
        if intent != "conversation":
            print(f"[INTENT: {intent}]")
        _brain_move = {
            "move_forward": "F", "move_backward": "B",
            "move_left": "L", "move_right": "R", "stop": "S",
        }
        if intent in _brain_move:
            cmd = _brain_move[intent]
            if self._movement_allowed(cmd):
                self.motors.move(cmd, None if cmd == "S" else 1.5)
        if emotion:
            self._apply_emotion(emotion)
        self.speak(reply)


