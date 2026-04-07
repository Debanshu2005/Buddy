"""
TiltController — single servo for up/down camera tilt.

Wiring:
  - Servo signal: GPIO 17
  - 50 Hz PWM, duty 2.5% (0°) → 12.5% (180°)

Angle range: 50° (down) → 130° (up), center = 90°
"""

import RPi.GPIO as GPIO
import time

SERVO_PIN  = 2
PWM_FREQ   = 50

TILT_MIN    = 50
TILT_MAX    = 130
TILT_CENTER = 90


def _duty(angle: float) -> float:
    return 2.5 + (angle / 180.0) * 10.0


class TiltController:
    def __init__(self):
        self._angle = TILT_CENTER
        self._pwm   = None

        try:
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(SERVO_PIN, GPIO.OUT)
            self._pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
            self._pwm.start(0)
            self._apply(TILT_CENTER)
            time.sleep(0.5)
            self._idle()
            print(f"✅ TiltController ready on GPIO{SERVO_PIN}")
        except Exception as e:
            print(f"⚠️  TiltController init failed: {e}")

    def move(self, angle: float):
        angle = max(TILT_MIN, min(TILT_MAX, angle))
        self._angle = angle
        self._apply(angle)

    def step(self, delta: float):
        self.move(self._angle + delta)

    def center(self):
        self.move(TILT_CENTER)

    @property
    def angle(self):
        return self._angle

    def cleanup(self):
        self.center()
        time.sleep(0.4)
        if self._pwm:
            self._pwm.stop()
        GPIO.cleanup(SERVO_PIN)
        print("🔌 TiltController cleaned up")

    def _apply(self, angle: float):
        if self._pwm:
            self._pwm.ChangeDutyCycle(_duty(angle))
            time.sleep(0.05)

    def _idle(self):
        if self._pwm:
            self._pwm.ChangeDutyCycle(0)
