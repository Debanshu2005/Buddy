"""Wake word mixin — Vosk wake word detection and sleep monitoring."""
from __future__ import annotations
import json, os, re, subprocess, threading
from typing import Optional

try:
    from vosk import KaldiRecognizer as _KaldiRecognizer
except Exception:
    _KaldiRecognizer = None


def _get_vosk_globals():
    """Fetch vosk state from buddy module at call time (loaded once there)."""
    if os.getenv("BUDDY_DISABLE_VOSK_WAKE", "1") != "0":
        return False, None
    try:
        import pi_runtime.buddy as _buddy
        return _buddy._vosk_available, _buddy._vosk_model
    except Exception:
        return False, None

class WakeMixin:
    def _is_ok_response(self, text: str) -> bool:
        lowered = text.lower().strip()
        return any(phrase in lowered for phrase in self._OK_RESPONSES)


    def _is_not_ok_response(self, text: str) -> bool:
        lowered = text.lower().strip()
        return any(phrase in lowered for phrase in self._NOT_OK_RESPONSES)


    def _is_emergency_phrase(self, text: str) -> bool:
        lowered = text.lower().strip()
        return any(phrase in lowered for phrase in self._EMERGENCY_PHRASES)


    def _normalize_heard_text(self, text: str) -> str:
        """Normalize recognition text before matching wake/emergency phrases."""
        lowered = text.lower().strip()
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered


    def _heard_wake_word(self, text: str) -> bool:
        """Return True when a wake phrase or common close variant is recognized."""
        normalized = self._normalize_heard_text(text)
        if not normalized:
            return False
        if len(normalized.split()) > 6:
            return False
        if any(phrase in normalized for phrase in self._WAKE_WORDS):
            return True
        tokens = normalized.split()
        if tokens == ["buddy"]:
            return True
        common_variants = ("budy", "baddi", "buddhi")
        return any(token in common_variants for token in tokens)


    def _extract_inline_command_from_wake(self, text: str) -> str:
        """Keep command words if the user says wake phrase and request together."""
        normalized = self._normalize_heard_text(text)
        if not normalized:
            return ""

        wake_phrases = sorted(self._WAKE_WORDS, key=len, reverse=True)
        for phrase in wake_phrases:
            if normalized == phrase:
                return ""
            if normalized.startswith(f"{phrase} "):
                return normalized[len(phrase):].strip()

        tokens = normalized.split()
        if not tokens:
            return ""
        if tokens[0] in {"buddy", "budy", "baddi", "buddhi"}:
            return " ".join(tokens[1:]).strip()
        return ""


    def _vosk_listen_for_wake_word(self) -> bool:
        _vosk_available, _vosk_model = _get_vosk_globals()
        if not _vosk_available:
            return False
        rec = _KaldiRecognizer(_vosk_model, 16000)
        wake_detected = False
        emergency_text = ""
        self._last_wake_text = ""
        proc = subprocess.Popen(
            ["arecord", "-D", self.arecord_device, "-f", "S16_LE", "-r", "16000", "-c", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        print(f"[Vosk] Listening on {self.arecord_device}...")
        try:
            while self.running:
                if self._pause_wake_listening:
                    return False
                raw = proc.stdout.read(8000)
                if not raw:
                    print("[Vosk] arecord stream ended")
                    break
                if self.is_speaking:
                    rec.Reset()
                    continue
                if rec.AcceptWaveform(raw):
                    text = json.loads(rec.Result()).get("text", "").lower()
                else:
                    text = json.loads(rec.PartialResult()).get("partial", "").lower()
                if not text:
                    continue
                normalized = self._normalize_heard_text(text)
                if self._is_emergency_phrase(normalized):
                    emergency_text = text
                    print(f"[Vosk] Emergency phrase: '{text}'")
                    break
                if self._heard_wake_word(normalized):
                    wake_detected = True
                    self._last_wake_text = normalized
                    print(f"Wake word: '{text}'")
                    break
        finally:
            proc.kill()
            proc.wait()

        if emergency_text:
            self._handle_passive_emergency_phrase("Vosk", emergency_text)
            return False

        return wake_detected


    def _vosk_monitor_sleep(self) -> bool:
        """
        Continuous Vosk listener for sleep/monitoring mode.
        - Stays running on a single arecord stream (no reconnect overhead).
        - Triggers emergency response immediately for any emergency phrase.
        - Returns True when wake word detected, False if stream ends.
        """
        _vosk_available, _vosk_model = _get_vosk_globals()
        if not _vosk_available:
            return False

        rec = _KaldiRecognizer(_vosk_model, 16000)
        self._last_wake_text = ""
        proc = subprocess.Popen(
            ["arecord", "-D", self.arecord_device, "-f", "S16_LE", "-r", "16000", "-c", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        print(f"[Monitor] Vosk monitoring on {self.arecord_device} — listening for emergencies and wake word...")

        try:
            while self.running and self.sleep_mode:
                raw = proc.stdout.read(8000)
                if not raw:
                    print("[Monitor] arecord stream ended")
                    break

                if self.is_speaking:
                    rec.Reset()
                    continue

                if rec.AcceptWaveform(raw):
                    text = json.loads(rec.Result()).get("text", "").lower()
                else:
                    text = json.loads(rec.PartialResult()).get("partial", "").lower()

                if not text:
                    continue

                normalized = self._normalize_heard_text(text)

                # Emergency — handle immediately, stay in monitoring mode
                if self._is_emergency_phrase(normalized):
                    print(f"[Monitor] 🚨 Emergency phrase: '{text}'")
                    threading.Thread(
                        target=self._handle_passive_emergency_phrase,
                        args=("Vosk", text),
                        daemon=True,
                    ).start()
                    rec.Reset()
                    continue

                # Wake word — exit monitoring mode
                if self._heard_wake_word(normalized):
                    self._last_wake_text = normalized
                    print(f"[Monitor] Wake word: '{text}'")
                    return True

        finally:
            proc.kill()
            proc.wait()

        return False



