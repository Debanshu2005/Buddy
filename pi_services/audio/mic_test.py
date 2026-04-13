import sounddevice as sd
import numpy as np

SAMPLE_RATE = 48000
DURATION = 3
CHANNELS = 2
DEVICE = 0   # ← THIS IS THE FIX

print("Recording...")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16",
    device=DEVICE
)
sd.wait()

print("Done.")
print("Max level:", np.max(np.abs(audio)))
