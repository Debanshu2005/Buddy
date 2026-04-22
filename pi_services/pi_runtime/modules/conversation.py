"""Conversation mixin — wake interaction, loops, response timing."""
from __future__ import annotations
import json, os, threading, time
from typing import Optional
from hardware.oled_eyes import EyeState

class ConversationMixin:
    def _response_user_key(self) -> str:
        return (self.active_user or "unknown").strip().lower() or "unknown"


    def _load_response_time_stats(self):
        try:
            if not self._response_time_stats_path.exists():
                return
            with open(self._response_time_stats_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                return
            for user_key, stats in data.items():
                if not isinstance(user_key, str) or not isinstance(stats, dict):
                    continue
                self._response_time_stats[user_key] = {
                    "avg": float(stats.get("avg", 0.0)),
                    "count": float(stats.get("count", 0.0)),
                    "last": float(stats.get("last", 0.0)),
                    "voice_avg_rms": float(stats.get("voice_avg_rms", 0.0)),
                    "voice_count": float(stats.get("voice_count", 0.0)),
                    "last_voice_rms": float(stats.get("last_voice_rms", 0.0)),
                    "last_voice_peak": float(stats.get("last_voice_peak", 0.0)),
                    "last_voice_duration": float(stats.get("last_voice_duration", 0.0)),
                }
        except Exception as exc:
            self.logger.warning("Could not load response timing stats: %s", exc)


    def _save_response_time_stats(self):
        try:
            self._response_time_stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._response_time_stats_path, "w", encoding="utf-8") as handle:
                json.dump(self._response_time_stats, handle, indent=2)
        except Exception as exc:
            self.logger.warning("Could not save response timing stats: %s", exc)


    def _record_response_time(self, elapsed: float):
        user_key = self._response_user_key()
        stats = self._response_time_stats.setdefault(
            user_key,
            {"avg": 0.0, "count": 0.0, "last": 0.0},
        )
        count = int(stats["count"])
        if count == 0:
            avg = elapsed
        else:
            avg = (stats["avg"] * (1.0 - self._RESPONSE_AVG_ALPHA)) + (elapsed * self._RESPONSE_AVG_ALPHA)
        stats.update({"avg": avg, "count": float(count + 1), "last": elapsed})
        print(f"[Response Watch] {user_key}: response={elapsed:.1f}s avg={avg:.1f}s samples={count + 1}")
        self._save_response_time_stats()


    def _response_is_delayed(self, elapsed: float) -> tuple[bool, float, int]:
        stats = self._response_time_stats.get(self._response_user_key())
        if not stats:
            return False, 0.0, 0
        avg = float(stats.get("avg", 0.0))
        count = int(stats.get("count", 0))
        if count < self._RESPONSE_DELAY_MIN_SAMPLES or avg <= 0:
            return False, avg, count
        delayed = (
            elapsed >= avg + self._RESPONSE_DELAY_SECONDS
            or elapsed >= avg * self._RESPONSE_DELAY_RATIO
        )
        return delayed, avg, count


    def _record_voice_stats(self, voice_stats: dict[str, float]):
        rms = float(voice_stats.get("rms", 0.0))
        if rms <= 0:
            return

        user_key = self._response_user_key()
        stats = self._response_time_stats.setdefault(
            user_key,
            {"avg": 0.0, "count": 0.0, "last": 0.0},
        )
        count = int(stats.get("voice_count", 0))
        if count == 0:
            avg_rms = rms
        else:
            avg_rms = (float(stats.get("voice_avg_rms", 0.0)) * (1.0 - self._VOICE_AVG_ALPHA)) + (
                rms * self._VOICE_AVG_ALPHA
            )

        stats.update(
            {
                "voice_avg_rms": avg_rms,
                "voice_count": float(count + 1),
                "last_voice_rms": rms,
                "last_voice_peak": float(voice_stats.get("peak", 0.0)),
                "last_voice_duration": float(voice_stats.get("duration", 0.0)),
            }
        )
        print(f"[Voice Watch] {user_key}: rms={rms:.4f} avg={avg_rms:.4f} samples={count + 1}")
        self._save_response_time_stats()


    def _voice_is_unusually_weak(self, voice_stats: dict[str, float]) -> tuple[bool, float, float, int]:
        rms = float(voice_stats.get("rms", 0.0))
        if rms <= 0:
            return False, rms, 0.0, 0

        stats = self._response_time_stats.get(self._response_user_key())
        if not stats:
            return False, rms, 0.0, 0

        avg_rms = float(stats.get("voice_avg_rms", 0.0))
        count = int(stats.get("voice_count", 0))
        if count < self._VOICE_WEAK_MIN_SAMPLES or avg_rms <= 0:
            return False, rms, avg_rms, count

        return rms <= avg_rms * self._VOICE_WEAK_RATIO, rms, avg_rms, count


    def _check_weak_voice_and_confirm(self, voice_stats: dict[str, float]) -> bool:
        weak, rms, avg_rms, count = self._voice_is_unusually_weak(voice_stats)
        if not weak:
            return True

        user_name = self.active_user or "there"
        reason = f"Voice sounded weaker than usual for {user_name}: {rms:.4f} vs average {avg_rms:.4f}"
        self.logger.warning("%s over %s samples", reason, count)
        self.speak("Your voice sounds weaker than usual. Are you okay?")
        self._wait_for_tts()
        self._play_listen_beep()
        self._eye(EyeState.LISTENING)
        check_text = self.listen_for_speech_with_initial_timeout(10.0)
        self._eye(EyeState.IDLE)

        if check_text and self._is_ok_response(check_text):
            self.speak("Okay. I am glad you are alright.")
            return True

        if not check_text or self._is_not_ok_response(check_text):
            self._trigger_emergency_response(check_text or reason)
            return False

        self.speak("I am not sure I understood. I will stay alert. If you need help, say emergency.")
        return True


    def _handle_wake_interaction(self, inline_command: str = "") -> None:
        """Wake word detected — play beep, listen once via Whisper, process, return."""
        print("Wake word detected")
        self._pause_wake_listening = True
        try:
            if inline_command:
                # wake phrase already contained a command — process it directly
                self._process_input(inline_command)
                self._wait_for_tts()
                return

            time.sleep(0.2)
            self._play_listen_beep()
            self._eye(EyeState.LISTENING)
            user_text = self.listen_for_speech()
            self._eye(EyeState.IDLE)

            if not user_text:
                return

            self._process_input(user_text)
            self._wait_for_tts()

            with self._notif_lock:
                pending = bool(self._notif_queue)
            if pending:
                threading.Thread(target=self._flush_notifications, daemon=True).start()
        finally:
            self._pause_wake_listening = False


    def _handle_passive_emergency_phrase(self, source: str, text: str):
        self.follow_mode = False
        self.motors.stop()
        self._trigger_emergency_response(f"{source} heard: {text}")
        self._wait_for_tts()


    def _check_response_delay_and_confirm(self, elapsed: float) -> bool:
        delayed, avg, count = self._response_is_delayed(elapsed)
        if not delayed:
            return True

        user_name = self.active_user or "there"
        self.logger.warning(
            "Response delay detected for %s: %.1fs vs %.1fs average over %s samples",
            user_name,
            elapsed,
            avg,
            count,
        )
        self.speak("You took longer than usual to answer. Are you okay?")
        self._wait_for_tts()
        self._play_listen_beep()
        self._eye(EyeState.LISTENING)
        check_text = self.listen_for_speech_with_initial_timeout(10.0)
        self._eye(EyeState.IDLE)

        if check_text and self._is_ok_response(check_text):
            self.speak("Okay. I am glad you are alright.")
            return True

        if not check_text or self._is_not_ok_response(check_text):
            self._trigger_emergency_response(check_text or "No response to wellness check")
            return False

        self.speak("I am not sure I understood. I will stay alert. If you need help, say emergency.")
        return True


    def _text_mode_loop(self):
        """Kept for compatibility — keyboard_loop now handles text mode activation."""
        pass


    def _run_text_session(self):
        """Interactive text session — runs until user types 'exit' or 'voice mode'."""
        import sys
        while self.running:
            try:
                sys.stdout.write("[You] ")
                sys.stdout.flush()
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            text = line.strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit", "voice mode", "back"):
                print("[Text Mode] Returning to voice mode.")
                break

            # registration: ask for name/password via terminal
            lowered = text.lower()
            if any(p in lowered for p in ("register my face", "register face", "add my face", "save my face")):
                self._text_mode_active = True
                try:
                    self._text_register_face()
                finally:
                    self._text_mode_active = True  # keep active — still in text session
                continue

            if self._is_identity_check(lowered):
                self._check_and_greet_face()
                self._wait_for_tts()
                continue

            # normal brain call
            self._play_thinking_sound()
            response = self._call_brain(text, self.active_user)
            self._thinking_token = None
            self._display_response(response)
            self._wait_for_tts()


    def _text_register_face(self):
        """Full registration flow via terminal input."""
        import sys

        sys.stdout.write("[Registration] What is your name? ")
        sys.stdout.flush()
        try:
            name_text = input().strip()
        except (EOFError, KeyboardInterrupt):
            return
        name = self._extract_name(name_text)
        if not name:
            words = [w.strip(".,!?") for w in name_text.split() if w.isalpha() and len(w) > 1]
            name = words[0].title() if words else ""
        if not name:
            print("[Registration] Could not get name. Cancelled.")
            return
        print(f"[Registration] Name: {name}")

        sys.stdout.write("[Registration] Set a password (any word or phrase): ")
        sys.stdout.flush()
        try:
            password = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if not password:
            print("[Registration] No password entered. Cancelled.")
            return
        print(f"[Registration] Password set.")

        from memory.pi_memory import save_password
        save_password(name, password)
        self.active_user = name

        print(f"[Registration] Starting face scan for {name}. Look straight at the camera...")
        self.speak(f"Starting face scan for {name}. Please look straight at the camera.")
        self._wait_for_tts()
        self._do_scan_then_save(name)


    def _handle_typed_input(self, text: str):
        """Process terminal input through the same command path as speech."""
        normalized = self._normalize_heard_text(text)
        if not normalized:
            return

        if normalized in ("quit", "exit"):
            self.speak("Shutting down.")
            self.cleanup()
            return

        if self.sleep_mode and self._heard_wake_word(normalized):
            self.sleep_mode = False
            self._eye(EyeState.WAKING)
            self.speak("I am awake. What can I do for you?")
            self._wait_for_tts()
            return

        self._process_input(text)


    def _keyboard_loop(self):
        """Allow typed interaction from the terminal alongside voice input."""
        while self.running:
            if self._text_mode_active:
                time.sleep(0.1)
                continue
            try:
                text = input("\nYou> ").strip()
            except EOFError:
                time.sleep(0.2)
                continue
            except Exception as exc:
                self.logger.warning("Keyboard input failed: %s", exc)
                time.sleep(0.5)
                continue

            if not text:
                continue
            if self._text_mode_active:
                continue

            lowered = text.lower().strip()
            if lowered in ("buddy", "hey buddy", "hi buddy"):
                print("\n[Text Mode] Activated. Type your command (or 'exit' to return to voice mode):")
                self._text_mode_active = True
                try:
                    self._run_text_session()
                finally:
                    self._text_mode_active = False
                continue

            print(f"[Typed] {text}")
            self._handle_typed_input(text)


    def _start_keyboard_listener(self):
        if self._keyboard_thread and self._keyboard_thread.is_alive():
            return
        self._keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._keyboard_thread.start()


    def _wake_loop(self):
        print("Waiting for 'Buddy'...")
        while self.running and not self.sleep_mode:
            if self._pause_wake_listening:
                time.sleep(0.1)
                continue
            if _vosk_available:
                detected = self._vosk_listen_for_wake_word()
                if not detected:
                    continue
                inline_command = self._extract_inline_command_from_wake(self._last_wake_text)
            else:
                text = self.listen_for_speech()
                lowered = self._normalize_heard_text(text) if text else ""
                if lowered and self._is_emergency_phrase(lowered):
                    self._handle_passive_emergency_phrase("STT", lowered)
                    continue
                if not self._heard_wake_word(lowered):
                    continue
                inline_command = self._extract_inline_command_from_wake(lowered)

            self._handle_wake_interaction(inline_command)
            print("Waiting for 'Buddy'...")

        if self.sleep_mode and self.running:
            self._sleep_loop()


    def _sleep_loop(self):
        print("Entering monitoring mode...")
        self.speak("Going to sleep. I will keep watching. Say hey buddy to wake me up.")
        self._wait_for_tts()
        self._eye(EyeState.SLEEPING)
        self._last_motion_time = time.time()

        while self.running and self.sleep_mode:
            try:
                if _vosk_available:
                    # Single continuous stream — handles emergencies inline, returns on wake word
                    if self._vosk_monitor_sleep():
                        inline_command = self._extract_inline_command_from_wake(self._last_wake_text)
                        self.sleep_mode = False
                        self._eye(EyeState.WAKING)
                        self.speak("I am awake. What can I do for you?")
                        self._wait_for_tts()
                        self._handle_wake_interaction(inline_command)
                        break
                else:
                    # No Vosk — use Whisper only for emergency phrases in sleep
                    text = self.listen_for_speech()
                    lowered = self._normalize_heard_text(text) if text else ""
                    if lowered and self._is_emergency_phrase(lowered):
                        self._handle_passive_emergency_phrase("STT", lowered)
                        continue
                    if self._heard_wake_word(lowered):
                        inline_command = self._extract_inline_command_from_wake(lowered)
                        self.sleep_mode = False
                        self._eye(EyeState.WAKING)
                        self.speak("I am awake. What can I do for you?")
                        self._wait_for_tts()
                        self._handle_wake_interaction(inline_command)
                        break
            except Exception as exc:
                self.logger.warning("Sleep loop error: %s", exc)
                time.sleep(1.0)

        if not self.sleep_mode and self.running:
            self._wake_loop()


