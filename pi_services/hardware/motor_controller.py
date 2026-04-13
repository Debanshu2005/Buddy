"""
MotorController — Serial bridge between Raspberry Pi and Arduino motor driver.

Movement model:
  - ALL moves run indefinitely until explicitly stopped.
  - Only three things stop the robot:
      1. stop() called (voice "stop"/"halt", or explicit code)
      2. Arduino detects an obstacle (sends OBSTACLE over Serial)
      3. A duration was specified ("move forward for 3 seconds") → timed stop

Arduino protocol:
  Pi → Arduino:   F  B  L  R  S  W  N   (single char + newline)
  Arduino → Pi:   READY | OBSTACLE | WIGGLE_DONE | NOD_DONE
"""

import serial
import threading
import time
import os
import logging

logger = logging.getLogger(__name__)

_ARDUINO_CANDIDATES = [
    "/dev/ttyUSB0", "/dev/ttyUSB1",
    "/dev/ttyACM0", "/dev/ttyACM1",
]
_BAUD = 115200


def _find_arduino_port() -> str:
    for port in _ARDUINO_CANDIDATES:
        if os.path.exists(port):
            return port
    return _ARDUINO_CANDIDATES[0]


class MotorController:

    # ── Emotion → Arduino command ─────────────────────────────────────────────
    # W = wiggle (Arduino animation), N = nod (Arduino animation)
    # F/B/L/R = run continuously until stopped
    EMOTION_MOVES = {
        # Happy / positive → wiggle
        "happy":        "W",
        "excited":      "W",
        "joyful":       "W",
        "friendly":     "W",
        "playful":      "W",
        "cheerful":     "W",
        "delighted":    "W",
        "proud":        "W",
        "enthusiastic": "W",
        # Thinking / uncertain → think back-and-forth
        "thinking":     "T",
        "confused":     "T",
        "uncertain":    "T",
        "pondering":    "T",
        "thoughtful":   "T",
        # Curious → pan left-right (short, self-stopping)
        "curious":      "PAN",
        "interested":   "PAN",
        "inquisitive":  "PAN",
        "attentive":    "PAN",
        # Sad / negative → slow backward until stopped
        "sad":          "B",
        "unhappy":      "B",
        "disappointed": "B",
        "sorry":        "B",
        "apologetic":   "B",
        "melancholy":   "B",
        # Angry → back away until stopped
        "angry":        "B",
        "frustrated":   "B",
        "annoyed":      "B",
        "displeased":   "B",
        # Surprised → forward until stopped
        "surprised":    "F",
        "shocked":      "F",
        "amazed":       "F",
        "impressed":    "F",
    }

    OBSTACLE_COOLDOWN = 5.0

    def __init__(self, port: str = None, baud: int = _BAUD):
        self._lock               = threading.Lock()
        self._obstacle_callback  = None
        self._last_obstacle_time = 0.0
        self._clear_callback     = None   # called when Arduino reports CLEAR

        # Keep-alive loop state (re-sends command every 0.8s against serial glitches)
        self._current_cmd    = "S"
        self._keepalive_active = False
        self._keepalive_thread = None

        # Duration-stop timer (only used when user specifies "for X seconds")
        self._duration_timer = None

        self._ser     = None
        self._running = False

        port = port or _find_arduino_port()
        try:
            self._ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2.0)
            self._running = True
            threading.Thread(target=self._read_loop, daemon=True).start()
            self._start_keepalive()
            print(f"✅ MotorController connected on {port} @ {baud} baud")
        except Exception as e:
            print(f"⚠️ MotorController: could not open {port}: {e}")
            self._ser = None

    # ── Public API ────────────────────────────────────────────────────────────

    def move(self, cmd: str, duration: float = None):
        """
        Move in direction cmd ('F','B','L','R') indefinitely.
        If duration is given, auto-stop after that many seconds.
        This is the ONE move method — everything routes through here.
        """
        self._cancel_duration_timer()
        self._current_cmd = cmd
        self._send_raw(cmd)
        print(f"🤖 Move → {cmd!r}" + (f"  (stop in {duration}s)" if duration else "  (until stop/obstacle)"))

        if duration:
            self._duration_timer = threading.Timer(duration, self.stop)
            self._duration_timer.daemon = True
            self._duration_timer.start()

    def stop(self):
        """Stop all motors immediately. Cancels duration timer and keep-alive."""
        self._cancel_duration_timer()
        self._current_cmd = "S"
        self._send_raw("S")
        print("🛑 Motors stopped")

    def emotion_move(self, emotion: str):
        """Express an emotion physically. Non-blocking."""
        e = emotion.lower().strip()
        print(f"🎭 Emotion: '{e}'")

        cmd = self.EMOTION_MOVES.get(e)
        if cmd is None:
            print(f"🎭 No movement for '{e}' — skipping")
            return

        def _do():
            if cmd == "PAN":
                # Short left-right pan — self-stopping, doesn't affect main movement
                self._send_raw("L"); time.sleep(0.35)
                self._send_raw("R"); time.sleep(0.35)
                self._send_raw("S")
                print(f"🎭 Pan done for '{e}'")
            elif cmd in ("W", "N", "T"):
                # Arduino handles full animation, sends WIGGLE_DONE/NOD_DONE when done
                self._send_raw(cmd)
                print(f"🎭 Animation '{cmd}' sent for '{e}'")
            else:
                # Continuous directional move (sad=B, surprised=F etc.)
                # Runs until obstacle or voice stop — same as any other move
                self.move(cmd)
                print(f"🎭 Continuous '{cmd}' for emotion '{e}'")

        threading.Thread(target=_do, daemon=True).start()

    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def set_obstacle_callback(self, fn):
        self._obstacle_callback = fn

    def set_clear_callback(self, fn):
        """Register a callable fired when Arduino reports path is clear again."""
        self._clear_callback = fn

    def cleanup(self):
        try:
            self.stop()
        except Exception:
            pass
        self._running = False
        self._keepalive_active = False
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        print("🔌 MotorController disconnected")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _send_raw(self, cmd: str):
        """Write command + newline to Arduino (thread-safe)."""
        if not self._ser or not self._ser.is_open:
            print(f"⚠️ Motor '{cmd}' ignored — Serial not available")
            return
        with self._lock:
            try:
                self._ser.write(f"{cmd}\n".encode())
                self._ser.flush()
            except Exception as e:
                print(f"⚠️ Serial write error: {e}")

    def _start_keepalive(self):
        """
        Re-send current movement command every 0.8s to guard against serial glitches.
        Only sends movement commands (F/B/L/R) — never resends S, W, or N.
        W and N are one-shot animations; S is explicit stop.
        """
        self._keepalive_active = True
        _MOVEMENT_CMDS = {"F", "B", "L", "R"}

        def _loop():
            while self._keepalive_active and self._running:
                time.sleep(0.8)
                if self._keepalive_active and self._current_cmd in _MOVEMENT_CMDS:
                    self._send_raw(self._current_cmd)

        self._keepalive_thread = threading.Thread(target=_loop, daemon=True)
        self._keepalive_thread.start()

    def _cancel_duration_timer(self):
        if self._duration_timer and self._duration_timer.is_alive():
            self._duration_timer.cancel()
        self._duration_timer = None

    def _read_loop(self):
        """Read Arduino replies — handle OBSTACLE, CLEAR, animation-done."""
        while self._running:
            try:
                if self._ser and self._ser.in_waiting:
                    line = self._ser.readline().decode(errors="ignore").strip()
                    if not line:
                        continue

                    if line == "OBSTACLE":
                        # Arduino has stopped motors and set obstacleBlocked=true.
                        # We MUST set _current_cmd = "S" immediately so the
                        # keep-alive loop stops sending movement commands.
                        # Arduino will reject them anyway, but sending "S" is
                        # cleaner and confirms the stop to Arduino.
                        self._cancel_duration_timer()
                        self._current_cmd = "S"
                        self._send_raw("S")          # explicit stop confirmation
                        print("🚧 OBSTACLE — motors stopped, waiting for clear")
                        if self._obstacle_callback:
                            try:
                                self._obstacle_callback()
                            except Exception:
                                pass

                    elif line == "CLEAR":
                        # Arduino reports obstacle cleared — path is open again.
                        # Don't auto-resume movement; wait for a new voice command.
                        print("✅ Path clear — say a command to move again")
                        if self._clear_callback:
                            try:
                                self._clear_callback()
                            except Exception:
                                pass

                    elif line in ("WIGGLE_DONE", "NOD_DONE", "THINK_DONE"):
                        print(f"✅ Animation done: {line}")
                        self._current_cmd = "S"

                    elif line == "READY":
                        print("✅ Arduino ready")

                    else:
                        print(f"🚗 Arduino: {line}")
                else:
                    time.sleep(0.05)
            except Exception as e:
                if self._running:
                    logger.debug(f"Read loop: {e}")
                time.sleep(0.1)
