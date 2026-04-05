#!/usr/bin/env python3
import subprocess
import os
import re
import time
import audioop
import wave

WHISPER_BIN = "/home/debanshu05/whisper.cpp/build/bin/whisper-cli"
MODEL_PATH  = "/home/debanshu05/whisper.cpp/models/ggml-tiny.en-q5_1.bin"

RATE     = "16000"
CHANNELS = "1"
DURATION = "3"

BASE_SILENCE_THRESHOLD = 250   # 🔥 lowered
GAIN_MULTIPLIER = 2.0          # 🔥 amplify signal


def find_usb_alsa_device() -> str:
    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True, text=True, timeout=5
        )

        for line in result.stdout.splitlines():
            if "usb" in line.lower() and line.strip().lower().startswith("card"):
                card_part = line.strip().split(":")[0]
                card = card_part.split()[-1]

                dev_match = re.search(r"\bdevice\s+(\d+)\b", line, re.IGNORECASE)
                dev = dev_match.group(1) if dev_match else "0"

                return f"hw:{card},{dev}"
    except:
        pass

    return "default"


ALSA_DEVICE = find_usb_alsa_device()


def rms_of_wav(path: str) -> float:
    with wave.open(path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())

        # 🔥 amplify before RMS (like PC version)
        frames = audioop.mul(frames, 2, GAIN_MULTIPLIER)

        rms = audioop.rms(frames, 2)
        return rms


def listen():
    wav_path = "/tmp/stt.wav"

    try:
        # 1️⃣ Record
        subprocess.run(
            [
                "arecord",
                "-D", ALSA_DEVICE,
                "-f", "S16_LE",
                "-r", RATE,
                "-c", CHANNELS,
                "-d", DURATION,
                wav_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )

        # 2️⃣ RMS detection
        rms = rms_of_wav(wav_path)
        print("RMS:", rms)

        if rms < BASE_SILENCE_THRESHOLD:
            return None

        print("🔊 Speech detected")

        # 3️⃣ Whisper
        result = subprocess.run(
            [
                WHISPER_BIN,
                "-m", MODEL_PATH,
                "-f", wav_path,
                "-t", "2",
                "--no-timestamps",
                "--language", "en",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

        text = result.stdout.strip()
        return text if text else None

    finally:
        try:
            os.remove(wav_path)
        except FileNotFoundError:
            pass


# ===================== MAIN LOOP =====================
if __name__ == "__main__":
    print(f"Whisper optimized STT ready (device: {ALSA_DEVICE})")

    while True:
        try:
            start = time.time()
            out = listen()
            end = time.time()

            if out:
                print(f"\nYou said: {out}")
            else:
                print("(silence)")

            print(f"⏱ Total cycle time: {round(end-start,2)} sec\n")

        except KeyboardInterrupt:
            print("\nExiting cleanly.")
            break
        except Exception as e:
            print("Error:", e)
            time.sleep(1)
