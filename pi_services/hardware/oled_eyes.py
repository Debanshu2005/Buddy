"""
OLED Eye Animations for Buddy.
Display : NFP1315-61AY  — 1.3" 128x64 OLED, SH1106 controller, I2C
Requires: luma.oled  (pip install luma.oled)
"""
import threading
import time
from dataclasses import dataclass
from enum import Enum

from luma.core.interface.serial import i2c
from luma.oled.device import sh1106
from PIL import Image, ImageDraw


class EyeState(Enum):
    IDLE       = "idle"
    SLEEPING   = "sleeping"
    WAKING     = "waking"
    LISTENING  = "listening"
    THINKING   = "thinking"
    SPEAKING   = "speaking"
    HAPPY      = "happy"
    SURPRISED  = "surprised"


@dataclass
class _Eye:
    cx: int   # centre x
    cy: int   # centre y
    w: int    # half-width
    h: int    # half-height
    lid: float = 0.0   # 0 = open, 1 = fully closed


_LEFT  = _Eye(cx=36,  cy=32, w=22, h=18)
_RIGHT = _Eye(cx=92,  cy=32, w=22, h=18)
_W, _H = 128, 64


def _draw_eye(draw: ImageDraw.ImageDraw, eye: _Eye):
    x0 = eye.cx - eye.w
    y0 = eye.cy - eye.h
    x1 = eye.cx + eye.w
    y1 = eye.cy + eye.h
    draw.ellipse([x0, y0, x1, y1], outline="white", fill="white")

    # pupil
    pr = 6
    draw.ellipse(
        [eye.cx - pr, eye.cy - pr, eye.cx + pr, eye.cy + pr],
        fill="black",
    )

    # upper lid (closes downward)
    if eye.lid > 0:
        lid_h = int(eye.h * 2 * eye.lid)
        draw.rectangle([x0, y0, x1, y0 + lid_h], fill="black")

    # lower lid (closes upward) — half speed
    if eye.lid > 0:
        lower_h = int(eye.h * eye.lid)
        draw.rectangle([x0, y1 - lower_h, x1, y1], fill="black")


def _render(device, left: _Eye, right: _Eye):
    img = Image.new("1", (_W, _H), "black")
    draw = ImageDraw.Draw(img)
    _draw_eye(draw, left)
    _draw_eye(draw, right)
    device.display(img)


class OledEyes:
    """Thread-safe OLED eye animator."""

    def __init__(self, i2c_port: int = 1, i2c_address: int = 0x3C):
        serial = i2c(port=i2c_port, address=i2c_address)
        # SH1106 has a 132-wide internal RAM; luma handles the 2-pixel offset
        self._device = sh1106(serial, width=_W, height=_H)
        self._state = EyeState.IDLE
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_state(self, state: EyeState):
        with self._lock:
            self._state = state

    def stop(self):
        self._stop.set()
        self._device.cleanup()

    # ------------------------------------------------------------------ #
    #  Animation loop                                                      #
    # ------------------------------------------------------------------ #
    def _loop(self):
        blink_timer = 0.0
        blink_interval = 3.5
        blinking = False
        blink_lid = 0.0

        # speaking bob
        bob = 0
        bob_dir = 1

        while not self._stop.is_set():
            with self._lock:
                state = self._state

            left  = _Eye(cx=_LEFT.cx,  cy=_LEFT.cy,  w=_LEFT.w,  h=_LEFT.h)
            right = _Eye(cx=_RIGHT.cx, cy=_RIGHT.cy, w=_RIGHT.w, h=_RIGHT.h)

            if state == EyeState.SLEEPING:
                left.lid = right.lid = 0.95
                _render(self._device, left, right)
                time.sleep(0.1)
                continue

            if state == EyeState.WAKING:
                for lid in [0.9, 0.7, 0.5, 0.3, 0.1, 0.0]:
                    left.lid = right.lid = lid
                    _render(self._device, left, right)
                    time.sleep(0.07)
                with self._lock:
                    if self._state == EyeState.WAKING:
                        self._state = EyeState.IDLE
                continue

            if state == EyeState.THINKING:
                # eyes look up-left
                left.cx  -= 4;  left.cy  -= 5
                right.cx -= 4;  right.cy -= 5
                _render(self._device, left, right)
                time.sleep(0.12)
                continue

            if state == EyeState.LISTENING:
                # slightly wider eyes
                left.w  = right.w  = 24
                left.h  = right.h  = 20

            if state == EyeState.HAPPY:
                # squint — partial lid + upward shift
                left.lid = right.lid = 0.35
                left.cy  -= 3;  right.cy -= 3

            if state == EyeState.SURPRISED:
                left.w  = right.w  = 26
                left.h  = right.h  = 22

            if state == EyeState.SPEAKING:
                # gentle vertical bob
                bob += bob_dir
                if abs(bob) >= 3:
                    bob_dir *= -1
                left.cy  += bob
                right.cy += bob

            # --- blink ---
            blink_timer += 0.05
            if not blinking and blink_timer >= blink_interval:
                blinking = True
                blink_lid = 0.0
                blink_timer = 0.0

            if blinking:
                blink_lid = min(blink_lid + 0.25, 1.0)
                left.lid  = max(left.lid,  blink_lid)
                right.lid = max(right.lid, blink_lid)
                if blink_lid >= 1.0:
                    blinking = False

            _render(self._device, left, right)
            time.sleep(0.05)
