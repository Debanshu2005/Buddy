"""Follow mixin — follow mode, clap detection, ultrasonic, obstacle handling."""
from __future__ import annotations
import os, subprocess, threading, time
from typing import Optional
import cv2
import numpy as np
from hardware.oled_eyes import EyeState

try:
    from gpiozero import DistanceSensor
except Exception:
    DistanceSensor = None

class FollowMixin:
    def _init_servo(self):
        if not self.settings.use_servo or os.getenv("BUDDY_ENABLE_SERVO", "1") == "0":
            self.logger.info("Servo disabled by settings")
            return
        try:
            from hardware.servo_controller import ServoController
            self.servo = ServoController(move_on_start=False)
            self.servo_enabled = bool(getattr(self.servo, "_pwm", None))
            if self.servo_enabled:
                self._servo_face_tracking_enabled = True
                self.logger.info("Servo initialized — face tracking enabled")
            else:
                self.logger.warning("Servo controller loaded but PWM is unavailable")
        except Exception as exc:
            self.servo = None
            self.servo_enabled = False
            self.logger.warning("Servo unavailable: %s", exc)

    def _init_ultrasonic_sensor(self):
        if not self.settings.ultrasonic_enabled:
            self.logger.info("Ultrasonic stop sensor disabled")
            return
        if DistanceSensor is None:
            self.logger.warning("Ultrasonic sensor requested but gpiozero is unavailable")
            return
        try:
            self._ultrasonic_sensor = DistanceSensor(
                echo=self.settings.ultrasonic_echo_pin,
                trigger=self.settings.ultrasonic_trigger_pin,
                max_distance=self.settings.ultrasonic_max_distance_m,
            )
        except Exception as exc:
            self._ultrasonic_sensor = None
            self.logger.warning("Ultrasonic sensor unavailable: %s", exc)
            return

        self._ultrasonic_thread = threading.Thread(target=self._ultrasonic_loop, daemon=True)
        self._ultrasonic_thread.start()
        self.logger.info(
            "Ultrasonic stop sensor enabled on trigger GPIO %s / echo GPIO %s (stop < %.2fm)",
            self.settings.ultrasonic_trigger_pin,
            self.settings.ultrasonic_echo_pin,
            self.settings.ultrasonic_stop_distance_m,
        )


    def _ultrasonic_loop(self):
        while not self.running and not self._cleaned_up:
            time.sleep(0.05)
        log_timer = 0.0
        while self.running and not self._cleaned_up:
            sensor = self._ultrasonic_sensor
            if sensor is None:
                return
            try:
                distance_m = float(sensor.distance) * float(self.settings.ultrasonic_max_distance_m)
                self._ultrasonic_distance_m = distance_m
                self._set_ultrasonic_blocked(distance_m <= self.settings.ultrasonic_stop_distance_m)
                log_timer += self._ULTRASONIC_POLL_INTERVAL
                if log_timer >= 2.0:
                    log_timer = 0.0
                    print(f"[Ultrasonic] distance={distance_m * 100:.1f} cm  blocked={self._ultrasonic_blocked}")
            except Exception as exc:
                self.logger.debug("Ultrasonic read failed: %s", exc)
            time.sleep(self._ULTRASONIC_POLL_INTERVAL)


    def _set_ultrasonic_blocked(self, blocked: bool):
        if blocked == self._ultrasonic_blocked:
            return
        self._ultrasonic_blocked = blocked
        dist_cm = int(self._ultrasonic_distance_m * 100) if self._ultrasonic_distance_m is not None else "?"
        if blocked:
            print(f"[Ultrasonic] 🚧 BLOCKED — {dist_cm} cm")
            self._follow_last_command = "S"
            self.motors.stop()
            self._announce_proximity_stop(force=True)
        else:
            print(f"[Ultrasonic] ✅ CLEAR — {dist_cm} cm")
            if not self.is_speaking:
                self.speak("Path is clear.")


    def _announce_proximity_stop(self, force: bool = False):
        now = time.time()
        if not force and (now - self._ultrasonic_last_announce_time) < self._ULTRASONIC_ANNOUNCE_COOLDOWN:
            return
        if self.is_speaking:
            return
        self._ultrasonic_last_announce_time = now
        distance_m = self._ultrasonic_distance_m
        if distance_m is None:
            self.speak("Object too close. I have stopped.")
            return
        distance_cm = int(distance_m * 100)
        self.speak(f"Object too close at about {distance_cm} centimeters. I have stopped.")


    def _movement_allowed(self, cmd: str, *, announce: bool = True) -> bool:
        if cmd == "S":
            return True
        if self._ultrasonic_blocked:
            self._follow_last_command = "S"
            self.motors.stop()
            if announce:
                self._announce_proximity_stop()
            return False
        return True


    def _prepare_follow_face_lock(self) -> bool:
        if self.servo_enabled and self.servo:
            try:
                center_angle = float(getattr(self.servo, "POS_CENTER", 90.0))
                target_angle = min(180.0, center_angle + self._FOLLOW_FACE_LOOK_UP_DEGREES)
                self.servo.move_to(target_angle, True)
                time.sleep(self._FOLLOW_FACE_SCAN_SETTLE_SECONDS)
            except Exception as exc:
                self.logger.warning("Servo follow positioning failed: %s", exc)

        deadline = time.time() + self._FOLLOW_FACE_SCAN_TIMEOUT
        while time.time() < deadline:
            frame = self.last_frame
            if frame is None:
                ok, fresh = self._read_frame()
                frame = fresh if ok else None
            if frame is not None and self.local_vision_enabled and self.detector is not None:
                try:
                    faces = self.detector.detect(frame)
                except Exception as exc:
                    self.logger.warning("Follow face detection failed: %s", exc)
                    faces = []
                if faces:
                    self._set_face_results(faces)
                    return True
            time.sleep(0.12)

        self._set_face_results([])
        return False


    def _on_obstacle(self):
        if not self.is_speaking and not self._ultrasonic_blocked:
            self.speak("Obstacle detected. I have stopped.")


    def _on_clear_path(self):
        if not self.is_speaking:
            self.speak("Path is clear.")


    def _clap_detection_loop(self):
        """Background thread — listens for claps via RMS spike on the mic."""
        mic_rate = 48000
        chunk_secs = 0.02
        bytes_per_chunk = int(mic_rate * chunk_secs) * 2

        CLAP_THRESHOLD   = float(os.getenv("BUDDY_CLAP_THRESHOLD",   "0.15"))
        CLAP_MAX_CHUNKS  = int(os.getenv("BUDDY_CLAP_MAX_CHUNKS",    "4"))
        SILENCE_CHUNKS   = int(os.getenv("BUDDY_CLAP_SILENCE_CHUNKS", "5"))  # min silence before clap

        device = self._working_arecord_device or self.arecord_device
        print(f"[Clap] Detection started on {device} (threshold={CLAP_THRESHOLD})")

        while self.running and not self._cleaned_up:
            if self.sleep_mode or self.is_speaking or self.follow_mode:
                time.sleep(0.2)
                continue

            try:
                proc = subprocess.Popen(
                    ["arecord", "-D", device, "-f", "S16_LE", "-r", str(mic_rate), "-c", "1"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )

                pre_silence  = 0   # chunks of silence before spike
                spike_count  = 0   # chunks above threshold
                in_spike     = False

                while self.running and not self._cleaned_up:
                    if self.sleep_mode or self.is_speaking or self.follow_mode:
                        break

                    raw = proc.stdout.read(bytes_per_chunk)
                    if not raw or len(raw) < bytes_per_chunk:
                        break

                    chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    rms   = float(np.sqrt(np.mean(chunk ** 2)))

                    if rms >= CLAP_THRESHOLD:
                        if not in_spike:
                            if pre_silence >= SILENCE_CHUNKS:
                                # valid clap start
                                in_spike    = True
                                spike_count = 1
                            else:
                                # not enough silence before — ignore
                                pre_silence = 0
                        else:
                            spike_count += 1
                            if spike_count > CLAP_MAX_CHUNKS:
                                # too long — it's speech/noise, not a clap
                                in_spike    = False
                                spike_count = 0
                                pre_silence = 0
                    else:
                        if in_spike and spike_count <= CLAP_MAX_CHUNKS:
                            # spike ended quickly — confirmed clap!
                            now = time.time()
                            if now - self._clap_last_time >= self._clap_cooldown:
                                self._clap_last_time = now
                                print(f"[Clap] 👏 Detected! spike={spike_count} chunks, threshold={CLAP_THRESHOLD}")
                                threading.Thread(
                                    target=self._on_clap_detected,
                                    daemon=True,
                                ).start()
                        in_spike    = False
                        spike_count = 0
                        pre_silence = min(pre_silence + 1, SILENCE_CHUNKS + 10)

            except Exception as exc:
                self.logger.debug("Clap detection error: %s", exc)
            finally:
                try:
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass
            time.sleep(0.5)


    def _detect_face_in_frame(self) -> Optional[tuple[float, float, float, float]]:
        """Return (x, y, w, h) of largest face in current frame, or None."""
        frame = self.last_frame
        if frame is None:
            ok, frame = self._read_frame()
            if not ok or frame is None:
                return None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
            if len(faces) > 0:
                return tuple(max(faces, key=lambda b: b[2] * b[3]))
        except Exception:
            pass
        return None


    def _on_clap_detected(self):
        """Rotate to find the person who clapped — left sweep then right sweep."""
        if not self.motors.is_connected():
            return
        if self._ultrasonic_blocked or self._emergency_active:
            return

        print("[Clap] Scanning for face...")
        self._eye(EyeState.CURIOUS)

        # Step 1 — check if face already visible straight ahead
        face = self._detect_face_in_frame()
        if face is not None:
            x, y, w, h = face
            frame_w = self.last_frame.shape[1] if self.last_frame is not None else 640
            center_x = (x + w / 2.0) / frame_w
            print(f"[Clap] Face already visible at x={center_x:.2f} — centering")
            self._center_on_face(center_x)
            self._eye(EyeState.HAPPY)
            return

        # Step 2 — sweep left
        print("[Clap] Sweeping left...")
        found = self._clap_sweep("L", sweep_duration=1.2, check_interval=0.15)
        if found:
            self._eye(EyeState.HAPPY)
            return

        # Step 3 — sweep right (from center)
        print("[Clap] Sweeping right...")
        self.motors.move("R", 0.6)   # return roughly to center first
        time.sleep(0.7)
        found = self._clap_sweep("R", sweep_duration=1.2, check_interval=0.15)
        if found:
            self._eye(EyeState.HAPPY)
            return

        # Step 4 — return to center, give up
        self.motors.move("L", 0.6)
        time.sleep(0.7)
        self.motors.stop()
        print("[Clap] No face found after sweep")
        self._eye(EyeState.IDLE)


    def _clap_sweep(self, direction: str, sweep_duration: float, check_interval: float) -> bool:
        """Rotate in direction, checking for face every check_interval seconds.
        Stops and centers when face found. Returns True if face found."""
        self.motors.move_follow(direction)
        elapsed = 0.0
        while elapsed < sweep_duration:
            time.sleep(check_interval)
            elapsed += check_interval
            if self._ultrasonic_blocked:
                self.motors.stop()
                return False
            face = self._detect_face_in_frame()
            if face is not None:
                self.motors.stop()
                x, y, w, h = face
                frame_w = self.last_frame.shape[1] if self.last_frame is not None else 640
                center_x = (x + w / 2.0) / frame_w
                print(f"[Clap] Face found during {direction} sweep at x={center_x:.2f}")
                self._center_on_face(center_x)
                return True
        self.motors.stop()
        return False


    def _center_on_face(self, face_center_x: float):
        """Fine-tune rotation to center the face horizontally."""
        error = face_center_x - 0.5
        tolerance = 0.12
        if abs(error) < tolerance:
            return
        direction = "R" if error > 0 else "L"
        nudge_time = min(0.4, abs(error) * 0.8)
        self.motors.move_follow(direction)
        time.sleep(nudge_time)
        self.motors.stop()


    def _listen_for_stop_command(self):
        """Listen in background for stop/emergency during continuous movement using STT."""
        print("[Move] Listening for stop command with STT...")
        while self.running:
            if self.motors._current_cmd == "S" or self.sleep_mode:
                break
            if self.is_speaking:
                time.sleep(0.1)
                continue

            text = self.listen_for_speech_with_initial_timeout(1.5)
            lowered = self._normalize_heard_text(text) if text else ""
            if not lowered:
                continue

            print(f"[Move] Heard via STT: '{lowered}'")
            if self._is_stop_command(lowered) or self._is_emergency_phrase(lowered):
                print("[Move] Stop command detected")
                self.motors.stop()
                self.speak("Stopped.")
                if self._is_emergency_phrase(lowered):
                    self._trigger_emergency_response(lowered)
                break


    def _start_follow_mode(self):
        if not self.motors.is_connected():
            self.speak("I cannot follow right now because the motors are not connected.")
            return
        if self._ultrasonic_blocked:
            self._announce_proximity_stop(force=True)
            return
        if not self._prepare_follow_face_lock():
            self.follow_mode = False
            self._follow_last_command = "S"
            self.motors.stop()
            self.speak("Face not found. Please move backward so I can see your face.")
            return
        self.follow_mode = True
        self._follow_last_command = "S"
        self._follow_last_command_time = 0.0
        self._follow_last_seen_time = time.time()
        self.speak("Okay, I will follow you. Say stop when you want me to stop.")


    def _stop_follow_mode(self):
        self.follow_mode = False
        self._follow_last_command = "S"
        self.motors.stop()
        self.speak("Okay, I stopped following.")


    def _get_follow_target(self, frame: np.ndarray) -> Optional[tuple[float, float, float]]:
        frame_h, frame_w = frame.shape[:2]
        frame_area = float(max(1, frame_w * frame_h))
        last_faces = self._get_faces_snapshot()
        current_detections = self._get_detections_snapshot()

        # try full vision stack first
        if last_faces:
            x, y, w, h = max(last_faces, key=lambda box: box[2] * box[3])
            return ((x + w / 2.0) / frame_w, (w * h) / frame_area, float(w * h))

        # fallback: lightweight haar cascade directly (works without onnxruntime)
        if not self.local_vision_enabled:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                    return ((x + w / 2.0) / frame_w, (w * h) / frame_area, float(w * h))
            except Exception:
                pass

        # fallback: person detection box
        person_boxes = []
        for det in current_detections:
            if not isinstance(det, dict) or det.get("name") != "person":
                continue
            box = det.get("box") or det.get("bbox") or det.get("xyxy")
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            area = width * height
            if area > 0:
                person_boxes.append((x1, y1, width, height, area))

        if not person_boxes:
            return None

        x, y, w, h, area = max(person_boxes, key=lambda box: box[4])
        return ((x + w / 2.0) / frame_w, area / frame_area, area)


    def _send_follow_command(self, cmd: str, duration: Optional[float] = None):
        now = time.time()
        if cmd == self._follow_last_command and (now - self._follow_last_command_time) < self._FOLLOW_COMMAND_INTERVAL:
            return
        if not self._movement_allowed(cmd):
            return
        self._follow_last_command = cmd
        self._follow_last_command_time = now
        self.motors.move_follow(cmd)


    def _update_follow_mode(self, frame: np.ndarray):
        if not self.follow_mode or self.sleep_mode or self._emergency_active:
            return
        if not self.motors.is_connected():
            self.follow_mode = False
            self.motors.stop()
            return
        if self._ultrasonic_blocked:
            self._send_follow_command("S")
            return

        target = self._get_follow_target(frame)
        if target is None:
            if time.time() - self._follow_last_seen_time > self._FOLLOW_LOST_TIMEOUT:
                self._send_follow_command("S")
            return

        self._follow_last_seen_time = time.time()
        center_x, area_ratio, _ = target
        error_x = center_x - 0.5

        center_tolerance = float(os.getenv("BUDDY_FOLLOW_CENTER_TOLERANCE", self._FOLLOW_CENTER_TOLERANCE))
        target_area = float(os.getenv("BUDDY_FOLLOW_TARGET_AREA_RATIO", self._FOLLOW_TARGET_AREA_RATIO))
        too_close_area = float(os.getenv("BUDDY_FOLLOW_TOO_CLOSE_AREA_RATIO", self._FOLLOW_TOO_CLOSE_AREA_RATIO))

        if abs(error_x) > center_tolerance:
            self._send_follow_command("R" if error_x > 0 else "L", 0.25)
            return

        if area_ratio < target_area:
            self._send_follow_command("F", 0.45)
            return

        if area_ratio > too_close_area:
            self._send_follow_command("B", 0.25)
            return

        self._send_follow_command("S")


