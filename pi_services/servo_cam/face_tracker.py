"""
FaceTracker — tilt-only face tracking (up/down).
Uses EMA smoothing on face position to handle blurry/noisy USB cameras.
Simple proportional control instead of PID to avoid oscillation/vibration.
"""

import cv2
import time
import threading

from .servo_cam_controller import TiltController

# ── Camera ─────────────────────────────────────────────────────────────────── #
CAMERA_INDEX = 0
FRAME_W      = 320
FRAME_H      = 240
FPS          = 15

# ── Detection ──────────────────────────────────────────────────────────────── #
CASCADES = [
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml",
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",
    cv2.data.haarcascades + "haarcascade_profileface.xml",
]
MIN_FACE       = (30, 30)
SCALE_FACTOR   = 1.05
MIN_NEIGHBORS  = 2
MISS_TOLERANCE = 6  # frames before losing track

# ── Smoothing ──────────────────────────────────────────────────────────────── #
# EMA alpha: lower = smoother but slower, higher = faster but noisier
# 0.2 works well for blurry cameras
EMA_ALPHA = 0.35  # higher = faster response, lower = smoother

# ── Control ────────────────────────────────────────────────────────────────── #
Kp       = 0.06   # proportional gain — how much to move per pixel of error
DEADZONE = 25     # pixels — stop moving when face is this close to center
MAX_STEP = 10.0   # degrees — hard cap per tick to prevent jerky jumps
LOOP_HZ  = 20     # higher loop rate = tighter sync with face movement


class FaceTracker:
    def __init__(self, show_preview: bool = False):
        self.servo        = TiltController()
        self.show_preview = show_preview
        self._running     = False
        self._thread      = None
        self._locked      = False

        self._last_face   = None
        self._miss_count  = 0
        self._smooth_cy   = None  # EMA smoothed face center Y
        self._locked_face = None  # the face we are committed to tracking

        self._cascades = []
        for path in CASCADES:
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                self._cascades.append(c)
        if not self._cascades:
            raise RuntimeError("No cascade classifiers could be loaded")
        print(f"✅ Loaded {len(self._cascades)} cascade(s)")

        self._cam = cv2.VideoCapture(CAMERA_INDEX)
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self._cam.set(cv2.CAP_PROP_FPS,          FPS)
        if not self._cam.isOpened():
            raise RuntimeError(f"Could not open USB camera at index {CAMERA_INDEX}")

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
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
    def is_locked(self):
        return self._locked

    def _loop(self):
        interval = 1.0 / LOOP_HZ

        while self._running:
            t0 = time.time()

            ret, frame = self._cam.read()
            if not ret:
                time.sleep(0.1)
                continue

            all_faces = self._detect_all_faces(frame)

            # ── face locking: stick to one face once acquired ──────────────── #
            if self._locked_face is None:
                # not tracking anyone yet — pick the largest face
                if all_faces:
                    self._locked_face = max(all_faces, key=lambda f: f[2] * f[3])
                    self._miss_count  = 0
            else:
                # already tracking — find the closest face to last known position
                lx, ly, lw, lh = self._locked_face
                lcx = lx + lw // 2
                lcy = ly + lh // 2
                best, best_dist = None, 80  # max pixels to still count as same face
                for f in all_faces:
                    fx, fy, fw, fh = f
                    dist = ((fx + fw//2 - lcx)**2 + (fy + fh//2 - lcy)**2) ** 0.5
                    if dist < best_dist:
                        best, best_dist = f, dist
                if best is not None:
                    self._locked_face = best
                    self._miss_count  = 0
                else:
                    self._miss_count += 1
                    if self._miss_count > MISS_TOLERANCE:
                        self._locked_face = None
                        self._smooth_cy   = None

            detected = self._locked_face is not None and self._miss_count == 0

            # ── update tracking state ──────────────────────────────────────── #
            if self._locked_face is not None:
                self._last_face  = self._locked_face
            else:
                self._last_face  = None
                self._smooth_cy  = None

            face = self._last_face

            if face is not None:
                fx, fy, fw, fh = face
                raw_cy = fy + fh // 2

                # EMA smoothing — filters out jitter from blurry detections
                if self._smooth_cy is None:
                    self._smooth_cy = float(raw_cy)
                else:
                    self._smooth_cy = EMA_ALPHA * raw_cy + (1 - EMA_ALPHA) * self._smooth_cy

                err_y = self._smooth_cy - FRAME_H // 2  # + = face below center

                if abs(err_y) < DEADZONE:
                    # locked on — cut PWM to stop vibration
                    self._locked = True
                    self.servo._idle()
                    print(f"🔒 LOCKED  tilt={self.servo.angle:.1f}°")
                else:
                    self._locked = False
                    # simple proportional step, capped at MAX_STEP
                    step = self._clamp(Kp * err_y, -MAX_STEP, MAX_STEP)
                    self.servo.step(-step)
                    print(f"🎯 err_y={err_y:+.0f}px  step={-step:+.1f}°  tilt={self.servo.angle:.1f}°")

                if self.show_preview:
                    color = (0, 255, 0) if detected is not None else (0, 165, 255)
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, 2)
                    cy_int = int(self._smooth_cy)
                    cv2.circle(frame, (fx + fw//2, cy_int), 5, (255, 0, 0), -1)
            else:
                self._locked = False
                print("👀 searching...")

            if self.show_preview:
                cv2.drawMarker(frame, (FRAME_W//2, FRAME_H//2), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
                cv2.imshow("FaceTracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False
                    break

            time.sleep(max(0, interval - (time.time() - t0)))

    def _detect_all_faces(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)

        all_faces = []
        for cascade in self._cascades:
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=SCALE_FACTOR,
                minNeighbors=MIN_NEIGHBORS,
                minSize=MIN_FACE
            )
            if len(faces) > 0:
                all_faces.extend(faces)

        # also try flipped frame for the other profile direction
        flipped = cv2.flip(gray, 1)
        profile = self._cascades[-1]  # profileface cascade
        faces_flipped = profile.detectMultiScale(
            flipped,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE
        )
        if len(faces_flipped) > 0:
            # mirror x coordinates back
            for (fx, fy, fw, fh) in faces_flipped:
                all_faces.append((FRAME_W - fx - fw, fy, fw, fh))

        return self._deduplicate(all_faces)

    @staticmethod
    def _deduplicate(faces, thresh=40):
        """Remove overlapping detections from multiple cascades."""
        result = []
        for face in faces:
            x, y, w, h = face
            duplicate = False
            for ex, ey, ew, eh in result:
                if abs(x - ex) < thresh and abs(y - ey) < thresh:
                    duplicate = True
                    break
            if not duplicate:
                result.append(face)
        return result

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
