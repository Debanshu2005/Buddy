#!/usr/bin/env python3
import subprocess
import tempfile
import os
from audio_lock import audio_lock

PIPER_BIN = "/home/debanshu05/piper/piper"
MODEL     = "/home/debanshu05/piper/models/en_US-lessac-medium.onnx"
CONFIG    = "/home/debanshu05/piper/models/en_US-lessac-medium.onnx.json"

AUDIO_DEV = "default"   # bcm2835 Headphones

def speak(text: str):
    if not text.strip():
        return

    with audio_lock:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            subprocess.run(
                [
                    PIPER_BIN,
                    "-m", MODEL,
                    "-c", CONFIG,
                    "-f", wav_path
                ],
                input=text,
                text=True,
                check=True
            )

            subprocess.run(
                ["aplay", "-D", AUDIO_DEV, wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)


# quick test
if __name__ == "__main__":
    speak("Hello. Buddy is alive.")
