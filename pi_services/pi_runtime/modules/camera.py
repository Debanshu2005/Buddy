"""Camera mixin — init, frame loop, vision tasks, stream server."""
from __future__ import annotations
import os, shutil, subprocess, sys, threading, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Optional

_MODULE_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
import cv2
import numpy as np
from hardware.oled_eyes import EyeState

class CameraMixin:
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
        total = sum(len(v) for v in self.recognizer.known_faces.values())
        self.logger.info("Local vision ready — %d people, %d embeddings loaded", len(self.recognizer.known_faces), total)


    def _init_camera(self):
        self.csi_frame_path = "/tmp/csi_frame.jpg"
        self.csi_process = None
        self.use_csi = False
        self.cap = None

        # Try PC webcam stream first if IP is configured
        if self.settings.pc_camera_ip:
            if self._start_pc_stream():
                self.logger.info("Using PC webcam stream from %s", self.settings.pc_camera_ip)
                return
            self.logger.warning("PC stream unavailable, falling back to local camera")

        helper_path = _MODULE_ROOT_DIR / "hardware" / "csi_camera_helper.py"
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

        self.logger.warning("No local camera found — running in camera-less mode")
        self.cap = None


    def _start_pc_stream(self) -> bool:
        """Start background thread pulling frames from the PC MJPEG stream."""
        import urllib.request
        url = f"http://{self.settings.pc_camera_ip}:{self.settings.pc_camera_port}/video"
        try:
            # Quick connectivity check
            urllib.request.urlopen(url, timeout=3).close()
        except Exception as exc:
            self.logger.warning("PC stream check failed: %s", exc)
            return False

        self._pc_stream_active = True

        def _pull():
            import urllib.request
            while self._pc_stream_active:
                try:
                    stream = urllib.request.urlopen(url, timeout=10)
                    buf = b""
                    while self._pc_stream_active:
                        buf += stream.read(4096)
                        start = buf.find(b"\xff\xd8")
                        end = buf.find(b"\xff\xd9")
                        if start != -1 and end != -1:
                            jpg = buf[start:end + 2]
                            buf = buf[end + 2:]
                            frame = cv2.imdecode(
                                np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                            )
                            if frame is not None:
                                self.last_frame = frame
                except Exception as exc:
                    self.logger.warning("PC stream interrupted: %s — retrying in 2s", exc)
                    time.sleep(2.0)

        self._pc_stream_thread = threading.Thread(target=_pull, daemon=True)
        self._pc_stream_thread.start()
        # Wait up to 3s for first frame
        for _ in range(30):
            if self.last_frame is not None:
                return True
            time.sleep(0.1)
        self.logger.warning("PC stream connected but no frames received")
        return True  # Still return True — stream may just be slow to start


    def _read_frame(self):
        if self._pc_stream_active:
            frame = self.last_frame
            return (True, frame.copy()) if frame is not None else (False, None)
        if self.use_csi and os.path.exists(self.csi_frame_path):
            frame = cv2.imread(self.csi_frame_path)
            return (True, frame) if frame is not None else (False, None)
        if self.cap is not None:
            return self.cap.read()
        return False, None


    def _get_faces_snapshot(self) -> list:
        with self._vision_state_lock:
            return list(self.last_faces)


    def _get_detections_snapshot(self) -> list:
        with self._vision_state_lock:
            return list(self.current_detections)


    def _get_objects_snapshot(self) -> tuple[list[str], set[str]]:
        with self._vision_state_lock:
            return list(self.current_objects), set(self.persistent_objects)


    def _set_face_results(self, faces: list) -> None:
        with self._vision_state_lock:
            self.last_faces = list(faces)

        if faces:
            if self._last_logged_face_present is not True:
                print("Detected face")
                self._last_logged_face_present = True
            largest = self.detector.get_largest_face(faces)
            self.stability.update(largest)
            return

        if self._last_logged_face_present is not False:
            self._last_logged_face_present = False
        self.stability.reset()


    def _set_object_results(self, detections: list[dict]) -> None:
        current_objects = [det["name"] for det in detections if det["name"] != "person"]
        with self._vision_state_lock:
            self.current_detections = list(detections)
            self.current_objects = current_objects
            self.persistent_objects.update(current_objects)
            current_objects_key = tuple(sorted(set(self.current_objects)))

        if current_objects_key != self._last_logged_objects:
            if current_objects_key:
                print(f"Objects: {', '.join(current_objects_key)}")
            self._last_logged_objects = current_objects_key


    def _schedule_camera_tasks(self, frame: np.ndarray) -> None:
        if self.local_vision_enabled:
            self._submit_camera_task("objects", frame)
            self._submit_camera_task("faces", frame)
        if self._behavior_pipeline is not None:
            self._submit_camera_task("pose_behavior", frame)


    def _submit_camera_task(self, task_name: str, frame: np.ndarray) -> None:
        task = self._camera_tasks.get(task_name)
        if task is None or self.frame_count % task.interval_frames != 0:
            return

        should_start_worker = False
        with self._camera_task_lock:
            task.latest_frame = frame.copy()
            task.latest_frame_count = self.frame_count
            if not task.busy:
                task.busy = True
                should_start_worker = True

        if should_start_worker:
            threading.Thread(
                target=self._camera_task_worker,
                args=(task_name,),
                daemon=True,
                name=f"CameraTask-{task_name}",
            ).start()


    def _camera_task_worker(self, task_name: str) -> None:
        while self.running and not self._cleaned_up:
            with self._camera_task_lock:
                task = self._camera_tasks.get(task_name)
                if task is None:
                    return
                frame = task.latest_frame
                frame_count = task.latest_frame_count
                task.latest_frame = None

            if frame is None:
                with self._camera_task_lock:
                    task = self._camera_tasks.get(task_name)
                    if task is None:
                        return
                    if task.latest_frame is not None:
                        continue
                    if task is not None:
                        task.busy = False
                return

            try:
                runner = getattr(self, task.runner_name)
                runner(frame, frame_count)
            except Exception as exc:
                self.logger.warning("%s worker failed: %s", task_name, exc)


    def _run_object_detection_task(self, frame: np.ndarray, frame_count: int) -> None:
        if not self.local_vision_enabled or self.object_detector is None:
            return
        detections = self.object_detector.detect(frame)
        if detections and os.getenv("BUDDY_LOG_DETECTIONS", "0") == "1":
            print(f"[Frame {frame_count}] Objects: {[(d['name'], f"{d['confidence']:.0%}") for d in detections]}")
        self._set_object_results(detections)


    def _run_face_detection_task(self, frame: np.ndarray, frame_count: int) -> None:
        if not self.local_vision_enabled or self.detector is None:
            return
        faces = self.detector.detect(frame)
        if faces and os.getenv("BUDDY_LOG_DETECTIONS", "0") == "1":
            print(f"[Frame {frame_count}] Faces: {len(faces)} detected, largest={(max(faces, key=lambda f: f[2]*f[3])[2:4]) if faces else None}")
        self._set_face_results(faces)


    def _run_pose_behavior_task(self, frame: np.ndarray, frame_count: int) -> None:
        if self._behavior_pipeline is None:
            return
        result = self._behavior_pipeline.process_frame(frame)
        if result.get("frame_processed") and result.get("person_detected"):
            decision = result.get("decision", {})
            if decision.get("label") not in ("normal", None):
                print(
                    f"[Behavior] {decision.get('label')} | "
                    f"{decision.get('severity')} | {decision.get('reason')}"
                )


    def _run_camera_frame_tasks(self, frame: np.ndarray) -> None:
        self._schedule_camera_tasks(frame)

        if self.frame_count % 3 == 0:
            self._update_follow_mode(frame)
            self._update_eye_tracking(frame)
            self._update_servo_face_tracking(frame)

        if self.frame_count % 10 == 0:
            self._update_behavior_monitor(frame)
            self._update_bbox_fall_monitor(frame)
            self._update_blood_monitor(frame)


    def _force_face_recognition(self) -> Optional[str]:
        try:
            ok, frame = self._read_frame()
            if not ok or frame is None or not self.local_vision_enabled:
                return None
            faces = self.detector.detect(frame)
            if not faces:
                return None
            for (x, y, w, h) in faces:
                if w > 80 and h > 80:
                    name, conf = self.recognizer.recognize(frame[y:y + h, x:x + w])
                    if name != "Unknown" and conf > 0.4:
                        print(f"Recognized: {name}")
                        return name
        except Exception as exc:
            self.logger.warning("Force recognition error: %s", exc)
        return None


    def _process_frame(self, frame: np.ndarray):
        return self._draw_visualization(frame.copy())


    def _draw_visualization(self, frame: np.ndarray) -> np.ndarray:
        if not self.local_vision_enabled:
            status = "Local vision disabled on this Pi"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return frame

        current_detections = self._get_detections_snapshot()
        last_faces = self._get_faces_snapshot()

        if current_detections:
            frame = self.object_detector.draw_detections(frame, current_detections)

        for (x, y, w, h) in last_faces:
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

        if self._ultrasonic_blocked:
            distance_cm = int(self._ultrasonic_distance_m * 100) if self._ultrasonic_distance_m is not None else None
            if distance_cm is None:
                status = "Obstacle too close"
            else:
                status = f"Obstacle too close: {distance_cm} cm"
        elif self.follow_mode:
            status = "Follow mode active"
        else:
            status = f"Chatting with: {self.active_user}" if self.active_user else "Looking for people..."
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame


    def _update_servo_face_tracking(self, frame: np.ndarray):
        if not self._servo_face_tracking_enabled:
            return
        last_faces = self._get_faces_snapshot()
        if not self.servo_enabled or not self.servo or not last_faces:
            return

        now = time.time()
        interval = float(os.getenv("BUDDY_SERVO_FACE_UPDATE_INTERVAL", self._SERVO_FACE_UPDATE_INTERVAL))
        if now - self._servo_face_last_update < interval:
            return

        frame_h, _ = frame.shape[:2]
        if frame_h <= 0:
            return

        x, y, w, h = max(last_faces, key=lambda box: box[2] * box[3])
        face_center_y = y + (h / 2.0)
        y_ratio = face_center_y / frame_h
        error = y_ratio - 0.5

        tolerance = float(os.getenv("BUDDY_SERVO_FACE_CENTER_TOLERANCE", self._SERVO_FACE_CENTER_TOLERANCE))
        if abs(error) < tolerance:
            return

        gain = float(os.getenv("BUDDY_SERVO_FACE_GAIN", self._SERVO_FACE_GAIN))
        min_angle = float(os.getenv("BUDDY_SERVO_FACE_MIN_ANGLE", self._SERVO_FACE_MIN_ANGLE))
        max_angle = float(os.getenv("BUDDY_SERVO_FACE_MAX_ANGLE", self._SERVO_FACE_MAX_ANGLE))
        try:
            current_angle = float(self.servo.get_angle())
        except Exception:
            return

        new_angle = current_angle - (error * gain)
        new_angle = max(min_angle, min(max_angle, new_angle))
        if abs(new_angle - current_angle) < 1.0:
            return

        self._servo_face_last_update = now
        threading.Thread(target=self.servo.move_to, args=(new_angle, True), daemon=True).start()


    def _update_eye_tracking(self, frame: np.ndarray):
        """Drive OLED pupil position from the currently tracked face."""
        if not self.eyes:
            return

        now = time.time()
        if now - self._eye_track_last_update < self._EYE_TRACK_UPDATE_INTERVAL:
            return

        last_faces = self._get_faces_snapshot()
        if not last_faces:
            self.eyes.center_gaze()
            self._eye_track_last_update = now
            return

        frame_h, frame_w = frame.shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            return

        x, y, w, h = max(last_faces, key=lambda box: box[2] * box[3])
        error_x = ((x + (w / 2.0)) / frame_w) - 0.5
        error_y = ((y + (h / 2.0)) / frame_h) - 0.5
        gaze_dx = int(np.clip(error_x * 20.0, -8, 8))
        gaze_dy = int(np.clip(error_y * 16.0, -6, 6))
        self.eyes.set_gaze(gaze_dx, gaze_dy)
        self._eye_track_last_update = now


    def _start_stream_server(self, port: int = 8080):
        buddy = self

        class _StreamHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"<html><body><img src='/stream'></body></html>")
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while buddy.running:
                            with buddy._stream_lock:
                                jpg = buddy._stream_frame
                            if jpg is None:
                                time.sleep(0.05)
                                continue
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                            )
                            self.wfile.flush()
                            time.sleep(0.04)
                    except Exception:
                        pass
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *args):
                return

        class _ThreadedServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True

        def _run():
            server = _ThreadedServer(("0.0.0.0", port), _StreamHandler)
            buddy.logger.info("Camera stream at http://<pi-ip>:%s/", port)
            try:
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
            except Exception:
                ip = "<pi-ip>"
            print(f"📷 Camera stream: http://{ip}:{port}/")
            server.serve_forever()

        threading.Thread(target=_run, daemon=True).start()


    def _camera_loop(self):
        while self.running:

            ok, frame = self._read_frame()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            self.last_frame = frame.copy()
            self.frame_count += 1
            self._run_camera_frame_tasks(frame)
            processed = self._process_frame(frame) if self.frame_count % 5 == 0 else frame

            ok, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with self._stream_lock:
                    self._stream_frame = buf.tobytes()

            time.sleep(0.02)


