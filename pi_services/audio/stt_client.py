import asyncio
import websockets
import sounddevice as sd
import numpy as np
from scipy.signal import resample_poly

SERVER_IP = "192.168.0.169"
PORT = 8765

MIC_RATE = 48000        # Native mic rate (change if needed)
TARGET_RATE = 16000     # Whisper expected rate
chunk_duration = 3

async def send_audio():
    uri = f"ws://{SERVER_IP}:{PORT}"

    async with websockets.connect(uri) as websocket:
        print("Connected to STT server")

        while True:
            print("🎤 Recording...")

            audio = sd.rec(int(chunk_duration * MIC_RATE),
                           samplerate=MIC_RATE,
                           channels=1,
                           dtype='float32')
            sd.wait()

            audio = np.squeeze(audio)

            # 🔥 Resample 48k → 16k
            audio_16k = resample_poly(audio, TARGET_RATE, MIC_RATE)

            await websocket.send(audio_16k.astype(np.float32).tobytes())

            text = await websocket.recv()
            print("📝 Server said:", text)

asyncio.run(send_audio())
