"""Patch buddy_update_integrated_response_watch.py for conversation vs monitoring modes."""
import re

path = "buddy_update_integrated_response_watch.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

original = src

# ── 1. Replace _process_conversation_turn + _run_conversation_session ──────────
old_conv = re.search(
    r"    def _process_conversation_turn\(.*?def _handle_passive_emergency_phrase",
    src, re.DOTALL
)
assert old_conv, "Could not find _process_conversation_turn"

new_conv = '''    def _run_conversation_session(self, initial_text: str = "") -> None:
        """Pure conversation mode — listen, respond, repeat until silence or sleep."""
        idle_timeout = float(os.getenv("BUDDY_CONVERSATION_IDLE_TIMEOUT", self._CONVERSATION_IDLE_TIMEOUT))
        pending_text = initial_text.strip()

        while self.running and not self.sleep_mode:
            if pending_text:
                user_text = pending_text
                pending_text = ""
            else:
                print("[Conversation] Listening...")
                time.sleep(0.2)
                self._play_listen_beep()
                self._eye(EyeState.LISTENING)
                user_text = self.listen_for_speech_with_initial_timeout(idle_timeout)
                self._eye(EyeState.IDLE)
                if not user_text:
                    print("[Conversation] No follow-up heard. Returning to wake mode.")
                    break

            self._process_input(user_text)
            self._wait_for_tts()
            if self.sleep_mode:
                break

        with self._notif_lock:
            pending = bool(self._notif_queue)
        if pending:
            threading.Thread(target=self._flush_notifications, daemon=True).start()

    def _handle_passive_emergency_phrase'''

src = src[:old_conv.start()] + new_conv + src[old_conv.end() - len("    def _handle_passive_emergency_phrase"):]

# ── 2. Sleep mode: keep camera running + enable monitoring ─────────────────────
# Fix _camera_loop to NOT skip frames in sleep mode
src = src.replace(
    "    def _camera_loop(self):\n        while self.running:\n            if self.sleep_mode:\n                time.sleep(0.1)\n                continue\n",
    "    def _camera_loop(self):\n        while self.running:\n"
)

# Fix _update_behavior_monitor to run in sleep mode
src = src.replace(
    "        if self.sleep_mode or self._emergency_active:\n            return\n\n        try:\n            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n            gray = cv2.GaussianBlur",
    "        if self._emergency_active:\n            return\n\n        try:\n            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n            gray = cv2.GaussianBlur"
)

# Fix _update_bbox_fall_monitor to run in sleep mode
src = src.replace(
    "        if os.getenv(\"BUDDY_ENABLE_BBOX_FALL_DETECTION\", \"1\") == \"0\":\n            return\n        if self.sleep_mode or self._emergency_active:\n            return",
    "        if os.getenv(\"BUDDY_ENABLE_BBOX_FALL_DETECTION\", \"1\") == \"0\":\n            return\n        if self._emergency_active:\n            return"
)

# Fix _update_blood_monitor to run in sleep mode
src = src.replace(
    "        if os.getenv(\"BUDDY_ENABLE_BLOOD_HEURISTIC\", \"1\") == \"0\":\n            return\n        if self.sleep_mode or self._emergency_active:\n            return",
    "        if os.getenv(\"BUDDY_ENABLE_BLOOD_HEURISTIC\", \"1\") == \"0\":\n            return\n        if self._emergency_active:\n            return"
)

# ── 3. Fix _sleep_loop: announce monitoring mode + reset motion timer ──────────
src = src.replace(
    '    def _sleep_loop(self):\r\n        print("Entering sleep mode...")\r\n        self.speak("Going to sleep. Say hey buddy to wake me up.")',
    '    def _sleep_loop(self):\n        print("Entering monitoring mode...")\n        self.speak("Going to sleep. I will keep watching. Say hey buddy to wake me up.")'
)
# also handle \n version
src = src.replace(
    '    def _sleep_loop(self):\n        print("Entering sleep mode...")\n        self.speak("Going to sleep. Say hey buddy to wake me up.")',
    '    def _sleep_loop(self):\n        print("Entering monitoring mode...")\n        self.speak("Going to sleep. I will keep watching. Say hey buddy to wake me up.")'
)

# Insert motion timer reset + eye state after _wait_for_tts in _sleep_loop
src = src.replace(
    '    def _sleep_loop(self):\n        print("Entering monitoring mode...")\n        self.speak("Going to sleep. I will keep watching. Say hey buddy to wake me up.")\n        self._wait_for_tts()\n',
    '    def _sleep_loop(self):\n        print("Entering monitoring mode...")\n        self.speak("Going to sleep. I will keep watching. Say hey buddy to wake me up.")\n        self._wait_for_tts()\n        self._eye(EyeState.SLEEPING)\n        self._last_motion_time = time.time()\n'
)

# ── 4. Fix sleep entry in _process_input to not set eye state (sleep_loop does it) ──
src = src.replace(
    "        # 4. Sleep\n        if any(phrase in lowered for phrase in self._SLEEP_PHRASES):\n            self.sleep_mode = True\n            self._eye(EyeState.SLEEPING)\n            return",
    "        # 4. Sleep\n        if any(phrase in lowered for phrase in self._SLEEP_PHRASES):\n            self.sleep_mode = True\n            return"
)

assert src != original, "No changes were made — check patterns"

with open(path, "w", encoding="utf-8") as f:
    f.write(src)

print("Patch applied successfully.")
