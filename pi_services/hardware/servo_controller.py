"""
ServoController - GPIO PWM servo control for camera tilt.

Servo specs:
  - Standard hobby servo (SG90 or similar)
  - Connected to GPIO 18 (physical pin 12)
  - 50 Hz PWM, duty cycle 2.5% (0 deg) to 12.5% (180 deg)

Positions:
  - DOWN (looking at chest/table level): ~30 deg
  - CENTER (eye level): ~90 deg
  - UP (looking at face): ~150 deg
"""

import threading
import time

import RPi.GPIO as GPIO

SERVO_PIN = 18
PWM_FREQ = 50


def _angle_to_duty(angle: float) -> float:
    """Convert angle (0-180) to PWM duty cycle (0-100)."""
    return 2.5 + (angle / 180.0) * 10.0


class ServoController:
    """Thread-safe servo controller that tracks the latest requested target."""

    POS_DOWN = 30
    POS_CENTER = 90
    POS_UP = 150

    def __init__(self, move_on_start: bool = False):
        self._current_angle = float(self.POS_CENTER)
        self._target_angle = float(self.POS_CENTER)
        self._pwm = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._move_event = threading.Event()
        self._worker_thread = None

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(SERVO_PIN, GPIO.OUT)
            self._pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
            self._pwm.start(0)

            if move_on_start:
                self._set_angle_immediate(self.POS_CENTER)
                time.sleep(0.5)
                self._pwm.ChangeDutyCycle(0)

            self._worker_thread = threading.Thread(target=self._movement_worker, daemon=True)
            self._worker_thread.start()
            print(f"ServoController initialized on GPIO {SERVO_PIN}")
        except Exception as exc:
            print(f"ServoController init error: {exc}")
            self._pwm = None

    def move_to(self, angle: float, smooth: bool = True):
        """
        Move servo to target angle.
        If smooth=True, the worker moves toward the latest target continuously.
        If smooth=False, jump directly to the requested angle.
        """
        if not self._pwm:
            return

        angle = max(0.0, min(180.0, float(angle)))
        with self._lock:
            self._target_angle = angle

        if smooth:
            self._move_event.set()
        else:
            self._set_angle_immediate(angle)

    def look_up(self, smooth: bool = True):
        """Move to UP position."""
        print(f"Servo: looking UP ({self.POS_UP} deg)")
        self.move_to(self.POS_UP, smooth)

    def look_down(self, smooth: bool = True):
        """Move to DOWN position."""
        print(f"Servo: looking DOWN ({self.POS_DOWN} deg)")
        self.move_to(self.POS_DOWN, smooth)

    def look_center(self, smooth: bool = True):
        """Move to CENTER position."""
        print(f"Servo: looking CENTER ({self.POS_CENTER} deg)")
        self.move_to(self.POS_CENTER, smooth)

    def _set_angle_immediate(self, angle: float):
        """Set servo angle instantly."""
        if not self._pwm:
            return
        self._pwm.ChangeDutyCycle(_angle_to_duty(angle))
        self._current_angle = float(angle)
        time.sleep(0.18)
        self._pwm.ChangeDutyCycle(0)

    def _movement_worker(self):
        """Move the servo toward the latest target without dropping updates."""
        step_size = 1.5
        delay = 0.025

        while not self._stop_event.is_set():
            self._move_event.wait()
            if self._stop_event.is_set():
                break
            if not self._pwm:
                self._move_event.clear()
                continue

            while not self._stop_event.is_set():
                with self._lock:
                    target = self._target_angle
                delta = target - self._current_angle

                if abs(delta) <= 0.6:
                    self._pwm.ChangeDutyCycle(_angle_to_duty(target))
                    self._current_angle = float(target)
                    time.sleep(0.03)
                    with self._lock:
                        settled = abs(self._target_angle - self._current_angle) <= 0.6
                    if settled:
                        self._pwm.ChangeDutyCycle(0)
                        self._move_event.clear()
                        break
                    continue

                step = max(-step_size, min(step_size, delta))
                self._current_angle += step
                self._pwm.ChangeDutyCycle(_angle_to_duty(self._current_angle))
                time.sleep(delay)

    def get_angle(self) -> float:
        """Return current servo angle."""
        return float(self._current_angle)

    def cleanup(self):
        """Stop PWM and release GPIO."""
        self._stop_event.set()
        self._move_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)
        if self._pwm:
            self._pwm.stop()
        GPIO.cleanup(SERVO_PIN)
        print("ServoController cleaned up")
