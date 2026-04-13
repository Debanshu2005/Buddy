"""
OLED Eye Animation for Buddy.
Display : NFP1315-61AY  — 1.3" 128x64 OLED, SH1106 controller, I2C
One OLED = one eye, centered on screen, drawn in blue.
Requires: luma.oled  (pip install luma.oled)
"""
import threading
import time
from dataclasses import dataclass
from enum import Enum

from luma.core.interface.serial import i2c
from luma.oled.device import sh1106
from PIL import Image, ImageDraw

_W, _H = 128, 64
_CX, _CY = 64, 32
_BLUE = "blue"


class EyeState(Enum):
    IDLE      = "idle"
    SLEEPING  = "sleeping"
    WAKING    = "waking"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"
    HAPPY     = "happy"
    SURPRISED = "surprised"


@dataclass
class _Eye:
    cx: int   = _CX
    cy: int   = _CY
    w: int    = 30
    h: int    = 24
    lid: float = 0.0
    pupil_dx: int = 0
    pupil_dy: int = 0


def _draw_eye(draw: ImageDraw.ImageDraw, eye: _Eye):
    x0, y0 = eye.cx - eye.w, eye.cy - eye.h
    x1, y1 = eye.cx + eye.w, eye.cy + eye.h

    draw.ellipse([x0, y0, x1, y1], outline=_BLUE, fill=_BLUE)

    pr = 9
    px, py = eye.cx + eye.pupil_dx, eye.cy + eye.pupil_dy
    draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill="black")

    if eye.lid > 0:
        draw.rectangle([x0, y0, x1, y0 + int(eye.h * 2 * eye.lid)], fill="black")
    if eye.lid > 0:
        draw.rectangle([x0, y1 - int(eye.h * eye.lid), x1, y1], fill="black")


def _render(device, eye: _Eye):
    img = Image.new("RGB", (_W, _H), "black")
    draw = ImageDraw.Draw(img)
    _draw_eye(draw, eye)
    device.display(img.convert(device.mode))


class OledEye:
    """Single-eye animator for one SH1106 OLED."""

    def __init__(self, i2c_port: int = 1, i2c_address: int = 0x3C):
        serial = i2c(port=i2c_port, address=i2c_address)
        self._device = sh1106(serial, width=_W, height=_H)
        self._state  = EyeState.IDLE
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_state(self, state: EyeState):
        with self._lock:
            self._state = state

    def stop(self):
        self._stop.set()
        self._device.cleanup()

    def _loop(self):
        blink_timer    = 0.0
        blink_interval = 3.5
        blinking       = False
        blink_lid      = 0.0
        bob = 0
        bob_dir = 1

        while not self._stop.is_set():
            with self._lock:
                state = self._state

            eye = _Eye()

            if state == EyeState.SLEEPING:
                eye.lid = 0.95
                _render(self._device, eye)
                time.sleep(0.1)
                continue

            if state == EyeState.WAKING:
                for lid in [0.9, 0.7, 0.5, 0.3, 0.1, 0.0]:
                    eye.lid = lid
                    _render(self._device, eye)
                    time.sleep(0.07)
                with self._lock:
                    if self._state == EyeState.WAKING:
                        self._state = EyeState.IDLE
                continue

            if state == EyeState.THINKING:
                # squished eye + pupil sweeps left-right
                eye.h = 10
                sweep = int(12 * (time.time() % 1.0 - 0.5) * 2)
                eye.pupil_dx = max(-10, min(10, sweep))
                _render(self._device, eye)
                time.sleep(0.05)
                continue

            if state == EyeState.LISTENING:
                eye.w, eye.h = 33, 27

            if state == EyeState.HAPPY:
                eye.lid = 0.35
                eye.cy -= 4

            if state == EyeState.SURPRISED:
                eye.w, eye.h = 36, 30

            if state == EyeState.SPEAKING:
                bob += bob_dir
                if abs(bob) >= 3:
                    bob_dir *= -1
                eye.cy += bob

            # --- blink ---
            blink_timer += 0.05
            if not blinking and blink_timer >= blink_interval:
                blinking    = True
                blink_lid   = 0.0
                blink_timer = 0.0

            if blinking:
                blink_lid = min(blink_lid + 0.25, 1.0)
                eye.lid   = max(eye.lid, blink_lid)
                if blink_lid >= 1.0:
                    blinking = False

            _render(self._device, eye)
            time.sleep(0.05)


# Backwards-compatible alias so buddy_pi_integrated.py needs no changes
class OledEyes(OledEye):
    pass


if __name__ == "__main__":
    print("Testing single-eye OLED on NFP1315-61AY — Ctrl+C to stop")
    eye = OledEye()
    sequence = [
        (EyeState.IDLE,      "IDLE      — normal blinking",          3),
        (EyeState.LISTENING, "LISTENING — wider eye",                3),
        (EyeState.THINKING,  "THINKING  — squished, pupil sweeping", 4),
        (EyeState.SPEAKING,  "SPEAKING  — gentle bob",               3),
        (EyeState.HAPPY,     "HAPPY     — squint",                   3),
        (EyeState.SURPRISED, "SURPRISED — big eye",                  3),
        (EyeState.SLEEPING,  "SLEEPING  — closed",                   3),
        (EyeState.WAKING,    "WAKING    — open animation",           2),
    ]
    try:
        for state, label, duration in sequence:
            print(f"  {label}")
            eye.set_state(state)
            time.sleep(duration)
    except KeyboardInterrupt:
        pass
    finally:
        eye.stop()
        print("Done.")
