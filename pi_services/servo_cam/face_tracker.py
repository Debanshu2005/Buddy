"""
FaceTracker — tilt-only face tracking (up/down).

Tracks the vertical position of the largest detected face and moves
the tilt servo to keep it centered in frame using a PID controller.

  - Large vertical offset → fast servo movement
  - Small offset          → slow, precise correction
  - Within deadzone       → servo stops (locked on)

Usage:
    tracker = FaceTracker()
    tracker.start()
    ...
    tracker.stop()
"""

import cv2
import time
import threading

from .servo_cam_controller import TiltController

# ── Camera ─────────────────────────────────────────────────────────────────── #
CAMERA_INDEX = 0     # change if your USB camera isn't /dev/video0
FRAME_W      = 320
FRAME_H      = 240
FPS          = 15

# ── Detection ──────────────────────────────────────────────────────────────── #
CASCADE        = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
MIN_FACE       = (30, 30)
SCALE_FACTOR   = 1.05  # finer steps = more detections
MIN_NEIGHBORS  = 2
MISS_TOLERANCE = 5     # frames without detection before giving up

# ── PID gains ──────────────────────────────────────────────────────────────── #
Kp = 0.15   # increase for faster tracking
Ki = 0.001  # corrects slow drift
Kd = 0.02   # reduce oscillation/overshoot

# ── Behaviour ──────────────────────────────────────────────────────────────── #
DEADZONE       = 25    # pixels — no movement when face is within this range of center
MAX_STEP       = 15.0  # degrees — max servo movement per tick
LOOP_HZ        = 15
MOTION_BOOST   = 1.8   # extra multiplier when face is actively moving
MOTION_THRESH  = 8     # pixels — min movement between frames to count as motion


class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._integral  = 0.0
        self._prev_err  = 0.0
        self._prev_time = time.time()

    def compute(self, error: float) -> float:
        now = time.time()
        dt  = max(now - self._prev_time, 1e-4)
        self._integral += error * dt
        derivative      = (error - self._prev_err) / dt
        output          = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._prev_err  = error
        self._prev_time = now
        return output

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0


class FaceTracker:
    def __init__(self, show_preview: bool = False):
        self.servo        = TiltController()
        self.show_preview = show_preview
        self._running     = False
        self._thread      = None
        self._locked      = False
        self._pid        = PID(Kp, Ki, Kd)
        self._last_face  = None  # last known face position
        self._miss_count = 0     # consecutive missed frames
        self._prev_cy    = None  # previous face center y for motion direction
        self._direction  = 0     # +1 = moving down, -1 = moving up, 0 = still

        self._cascade = cv2.CascadeClassifier(CASCADE)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load cascade: {CASCADE}")

        self._cam = cv2.VideoCapture(CAMERA_INDEX)
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self._cam.set(cv2.CAP_PROP_FPS,          FPS)
        if not self._cam.isOpened():
            raise RuntimeError(f"Could not open USB camera at index {CAMERA_INDEX}")

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("🎯 FaceTracker started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._cam.release()
        self.servo.cleanup()
        if self.show_preview:
            cv2.destroyAllWindows()
        print("🛑 FaceTracker stopped")

    @property
    def is_locked(self) -> bool:
        return self._locked

    def _loop(self):
        interval = 1.0 / LOOP_HZ

        while self._running:
            t0  = time.time()
            ret, frame = self._cam.read()
            if not ret:
                time.sleep(0.1)
                continue
            detected = self._detect_largest_face(frame)

            if detected is not None:
                self._last_face  = detected
                self._miss_count = 0
            else:
                self._miss_count += 1
                if self._miss_count > MISS_TOLERANCE:
                    self._last_face = None

            face = self._last_face

            if face is not None:
                fx, fy, fw, fh = face
                face_cy = fy + fh // 2
                err_y   = face_cy - FRAME_H // 2

                # detect motion direction
                if self._prev_cy is not None:
                    dy = face_cy - self._prev_cy
                    if abs(dy) >= MOTION_THRESH:
                        self._direction = 1 if dy > 0 else -1
                    else:
                        self._direction = 0
                self._prev_cy = face_cy

                if abs(err_y) < DEADZONE and self._direction == 0:
                    self._locked = True
                    self._pid.reset()
                    self.servo._idle()
                    print(f"🔒 LOCKED  tilt={self.servo.angle:.1f}°")
                else:
                    self._locked = False
                    boost = MOTION_BOOST if self._direction != 0 else 1.0
                    step  = self._clamp(self._pid.compute(err_y) * boost, -MAX_STEP, MAX_STEP)
                    # also nudge immediately in motion direction even if err_y is small
                    if self._direction != 0 and abs(err_y) < DEADZONE:
                        step = -self._direction * 3.0
                    self.servo.step(-step)
                    dir_str = "⬆️" if self._direction == -1 else "⬇️" if self._direction == 1 else "➡️"
                    print(f"🎯 {dir_str}  err_y={err_y:+.0f}px  step={-step:+.1f}°  tilt={self.servo.angle:.1f}°")

                if self.show_preview:
                    color = (0, 255, 0) if detected is not None else (0, 165, 255)
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, 2)
                    cv2.circle(frame, (fx + fw//2, face_cy), 4, color, -1)
            else:
                self._locked = False
                self._prev_cy = None
                self._direction = 0
                self._pid.reset()
                print("👀 no face detected")

            if self.show_preview:
                cv2.drawMarker(frame, (FRAME_W//2, FRAME_H//2), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
                cv2.imshow("FaceTracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False
                    break

            time.sleep(max(0, interval - (time.time() - t0)))

    def _detect_largest_face(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS, minSize=MIN_FACE
        )
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    @staticmethod
    def _clamp(val, lo, hi):
        return max(lo, min(hi, val))


if __name__ == "__main__":
    tracker = FaceTracker(show_preview=True)
    try:
        tracker.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        tracker.stop()
