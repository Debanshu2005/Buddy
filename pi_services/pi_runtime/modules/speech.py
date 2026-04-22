"""Speech mixin — audio device discovery, VAD recording, Whisper STT, TTS, beep."""
from __future__ import annotations
import asyncio, os, subprocess, tempfile, threading, time, wave
from typing import Optional
import numpy as np
from scipy.signal import resample_poly
import edge_tts
from hardware.oled_eyes import EyeState

class SpeechMixin:
    def _scan_alsa_cards(self, tool: str) -> list[tuple[str, str, str]]:
        results = []
        try:
            result = subprocess.run([tool, "-l"], capture_output=True, text=True, timeout=5)
            for line in result.stdout.splitlines():
                if not line.startswith("card"):
                    continue
                try:
                    card_num = line.split(":")[0].strip().split()[-1]
                    dev_num = line.split("device")[-1].strip().split(":")[0].strip()
                    card_name = line.split(":")[1].split("[")[0].strip().replace(" ", "_")
                    results.append((card_num, card_name, dev_num))
                except Exception:
                    continue
        except Exception:
            pass
        return results


    def _find_output_audio_device(self) -> tuple[str, str]:
        cards = self._scan_alsa_cards("aplay")
        for card_num, card_name, dev_num in cards:
            if "usb" in card_name.lower() or "bcm2835" in card_name.lower() or "headphone" in card_name.lower():
                return f"plughw:{card_num},{dev_num}", card_num
        return "default", "0"


    def _find_input_audio_device(self) -> str:
        cards = self._scan_alsa_cards("arecord")
        print(f"[Audio] arecord cards: {cards}")
        if cards:
            card_num, card_name, dev_num = cards[0]
            device = f"plughw:{card_num},{dev_num}"
            print(f"[Audio] Using mic: {device} ({card_name})")
            return device
        return "default"


    def _generate_beep_wav(self) -> str:
        sample_rate = 22050
        duration = 0.14
        tones = [(740, duration), (920, duration)]
        frames = []
        for freq, dur in tones:
            t = np.linspace(0, dur, int(sample_rate * dur), endpoint=False)
            tone = self.listen_beep_gain * np.sin(2 * np.pi * freq * t)
            frames.append((tone * 32767).astype(np.int16).tobytes())

        wav_path = tempfile.mktemp(suffix=".wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
        return wav_path


    def _play_listen_beep(self):
        wav_path = None
        try:
            wav_path = self._generate_beep_wav()
            for device in (self.aplay_device, "default"):
                result = subprocess.run(
                    ["aplay", "-D", device, "-q", wav_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=4,
                )
                if result.returncode == 0:
                    break
        except Exception as exc:
            self.logger.warning("Listen beep failed: %s", exc)
        finally:
            if wav_path:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass


    def _eye(self, state: EyeState):
        if self.eyes:
            self.eyes.set_state(state)


    def speak(self, text: str):
        if not text.strip():
            return
        print(f"\nBuddy: {text}")
        self.is_speaking = True
        self._eye(EyeState.SPEAKING)

        def _run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._tts(text))
                loop.close()
            except Exception as exc:
                self.logger.warning("TTS failed: %s", exc)
            finally:
                self.is_speaking = False
                self._eye(EyeState.IDLE)

        threading.Thread(target=_run, daemon=True).start()

    async def _tts(self, text: str):
        communicate = edge_tts.Communicate(text, self.tts_voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        if not audio_data:
            return

        mp3_path = tempfile.mktemp(suffix=".mp3")
        wav_path = tempfile.mktemp(suffix=".wav")
        with open(mp3_path, "wb") as handle:
            handle.write(audio_data)
        try:
            result = subprocess.run(
                [
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    "-volume",
                    "100",
                    "-af",
                    f"volume={self.tts_gain}",
                    mp3_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
            if result.returncode != 0:
                raise Exception(f"ffplay returned {result.returncode}")
        except Exception as e:
            print(f"[TTS] ffplay failed ({e}), trying ffmpeg/aplay...")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "quiet", "-i", mp3_path, wav_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                    check=True,
                )
                subprocess.run(
                    ["aplay", "-D", self.aplay_device, "-q", wav_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                    check=True,
                )
            except Exception as e2:
                print(f"[TTS] ffmpeg/aplay also failed: {e2}")
        finally:
            try:
                os.remove(mp3_path)
            except OSError:
                pass
            try:
                os.remove(wav_path)
            except OSError:
                pass


    def _wait_for_tts(self, timeout: float = 30.0):
        start = time.time()
        while self.is_speaking and (time.time() - start) < timeout:
            time.sleep(0.05)
        time.sleep(0.05)


    def _record_audio_vad(self) -> np.ndarray:
        mic_rate = 48000
        target_chunk_secs = 0.1
        speech_thresh = 0.005
        silence_after = 0.4
        min_speech = 0.1
        max_duration = float(os.getenv("BUDDY_RESPONSE_LISTEN_MAX", "8.0"))
        initial_wait_timeout = self._listen_initial_timeout

        # use cached device — skip probe after first successful use
        if self._working_arecord_device:
            working = self._working_arecord_device
        else:
            candidates = [self.arecord_device, "default"]
            working = None
            for device in candidates:
                try:
                    probe = tempfile.mktemp(suffix=".wav")
                    result = subprocess.run(
                        ["arecord", "-D", device, "-f", "S16_LE", "-r", str(mic_rate), "-c", "1", "-d", "1", probe],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=4,
                    )
                    try:
                        os.remove(probe)
                    except OSError:
                        pass
                    if result.returncode == 0:
                        working = device
                        self._working_arecord_device = device
                        print(f"🎤 VAD using device: {device}")
                        break
                except Exception:
                    continue

        if not working:
            return np.array([], dtype=np.float32)

        bytes_per_chunk = int(mic_rate * target_chunk_secs) * 2
        proc = subprocess.Popen(
            ["arecord", "-D", working, "-f", "S16_LE", "-r", str(mic_rate), "-c", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        speech_started = False
        silence_duration = 0.0
        speech_duration = 0.0
        total_duration = 0.0
        chunks: list[np.ndarray] = []

        try:
            while total_duration < max_duration:
                raw = proc.stdout.read(bytes_per_chunk)
                if not raw or len(raw) < bytes_per_chunk:
                    break
                chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                total_duration += target_chunk_secs

                if (
                    initial_wait_timeout is not None
                    and not speech_started
                    and total_duration >= initial_wait_timeout
                ):
                    break

                if rms >= speech_thresh:
                    if not speech_started:
                        print("🎤️ Speech detected")
                    speech_started = True
                    silence_duration = 0.0
                    speech_duration += target_chunk_secs
                    chunks.append(chunk)
                elif speech_started:
                    silence_duration += target_chunk_secs
                    chunks.append(chunk)
                    if silence_duration >= silence_after:
                        break
        finally:
            proc.kill()
            proc.wait()

        if not speech_started or speech_duration < min_speech:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    async def _ws_listen_once(self) -> str:
        import websockets

        audio = await asyncio.get_event_loop().run_in_executor(None, self._record_audio_vad)
        self._last_voice_stats = self._measure_voice_stats(audio)
        if audio.size == 0:
            return ""

        audio_16k = resample_poly(audio, 16000, 48000).astype(np.float32)
        max_samples = 16000 * 8
        if len(audio_16k) > max_samples:
            audio_16k = audio_16k[:max_samples]
        stt_host = (self.settings.stt_server_ip or "").strip()
        if stt_host in {"0.0.0.0", "::", "[::]"}:
            stt_host = "127.0.0.1"
            if not self._stt_endpoint_warned:
                self._stt_endpoint_warned = True
                self.logger.warning(
                    "BUDDY_STT_SERVER_IP was set to a wildcard address; using %s for client connection.",
                    stt_host,
                )
        uri = f"ws://{stt_host}:{self.settings.stt_port}"
        print(f"[STT] Connecting to {uri} ({len(audio_16k)/16000:.1f}s audio)")
        try:
            async with websockets.connect(
                uri,
                ping_interval=None,
                open_timeout=10,
                close_timeout=5,
            ) as websocket:
                print("[STT] Connected — sending audio")
                await websocket.send(audio_16k.tobytes())
                print("[STT] Audio sent — waiting for transcription")
                result = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                print(f"[STT] Received: '{result.strip()}'")
                return result.strip() if result else ""
        except asyncio.TimeoutError:
            self.logger.warning("[STT] Timed out waiting for transcription from %s", uri)
            return ""
        except OSError as exc:
            self.logger.warning("[STT] Cannot reach server at %s — %s", uri, exc)
            return ""
        except Exception as exc:
            self.logger.warning("[STT] Websocket failed: %s", exc)
            return ""


    def _measure_voice_stats(self, audio: np.ndarray) -> dict[str, float]:
        if audio.size == 0:
            return {}
        abs_audio = np.abs(audio)
        return {
            "rms": float(np.sqrt(np.mean(audio ** 2))),
            "peak": float(np.max(abs_audio)),
            "duration": float(audio.size / 48000.0),
        }


    def listen_for_speech(self) -> str:
        print("🎤 Listening (VAD)...")
        try:
            if self._listen_loop is None or self._listen_loop.is_closed():
                self._listen_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._listen_loop)
            text = self._listen_loop.run_until_complete(self._ws_listen_once())
            if text and len(text) > 1:
                print(f"🎤 Heard: '{text}'")
                return text
            print("🔇 No speech detected")
            return ""
        except Exception as exc:
            self.logger.warning("listen_for_speech failed: %s", exc)
            self._listen_loop = None
            return ""


    def listen_for_speech_with_initial_timeout(self, timeout: float) -> str:
        previous_timeout = self._listen_initial_timeout
        self._listen_initial_timeout = timeout
        try:
            return self.listen_for_speech()
        finally:
            self._listen_initial_timeout = previous_timeout


    def _startup_greeting(self):
        time.sleep(1.0)
        self.speak("Hey. I am Buddy. Call me anytime.")


