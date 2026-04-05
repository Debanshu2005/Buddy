import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json
import numpy as np
import samplerate
import queue
import threading
import sys

MODEL_PATH = "vosk-model-small-en-us-0.15"
DEVICE_INDEX = 1
MIC_RATE = 48000
VOSK_RATE = 16000

audio_queue = queue.Queue()

print("Loading model...")
model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, VOSK_RATE)
rec.SetWords(False)
rec.SetMaxAlternatives(0)

print("🎤 Vosk ready. Speak now...")

# 🔹 Audio callback (DO NOTHING HEAVY HERE)
def audio_callback(indata, frames, time, status):
    if status:
        print("Audio:", status, file=sys.stderr)
    audio_queue.put(indata.copy())

# 🔹 Worker thread (processing happens here)
def worker():
    while True:
        data = audio_queue.get()

        # Convert to float
        audio_float = data[:, 0].astype(np.float32) / 32768.0

        # Downsample
        audio_16k = samplerate.resample(
            audio_float,
            VOSK_RATE / MIC_RATE,
            'sinc_fastest'
        )

        audio_16k = (audio_16k * 32768).astype(np.int16)

        if rec.AcceptWaveform(audio_16k.tobytes()):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            if text:
                print("You said:", text)

# Start worker thread
threading.Thread(target=worker, daemon=True).start()

with sd.InputStream(
    device=DEVICE_INDEX,
    samplerate=MIC_RATE,
    blocksize=8192,
    dtype="int16",
    channels=1,
    latency='high',
    callback=audio_callback):

    while True:
        pass
