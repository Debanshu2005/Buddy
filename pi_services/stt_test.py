#!/usr/bin/env python3
"""
Direct test of STT pipeline - bypasses all Buddy code
Tests: arecord → resample → WebSocket → Whisper → response
"""
import asyncio
import websockets
import subprocess
import tempfile
import numpy as np
import wave
import time
from scipy.signal import resample_poly

# CHANGE THESE TO MATCH YOUR SETUP
STT_SERVER_IP = "192.168.0.169"
STT_PORT = 8765
MIC_DEVICE = "plughw:3,0"  # Your USB mic
MIC_RATE = 48000
TARGET_RATE = 16000
DURATION = 5  # seconds

async def test_stt():
    print("=" * 60)
    print("STT PIPELINE TEST")
    print("=" * 60)
    
    # Step 1: Record
    print(f"\n1️⃣ Recording {DURATION}s from {MIC_DEVICE}...")
    wav_file = tempfile.mktemp(suffix=".wav")
    cmd = [
        "arecord", "-D", MIC_DEVICE,
        "-f", "S16_LE", "-r", str(MIC_RATE), "-c", "1",
        "-d", str(DURATION), wav_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Recording failed: {result.stderr}")
        return
    print("✅ Recording complete")
    
    # Step 2: Load and check RMS
    print("\n2️⃣ Loading and analyzing audio...")
    with wave.open(wav_file, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        samples /= 32768.0
    
    rms = float(np.sqrt(np.mean(samples ** 2)))
    print(f"   RMS (no gain): {rms:.6f}")
    
    # Apply gain
    GAIN = 4.0
    samples = np.clip(samples * GAIN, -1.0, 1.0)
    rms_gained = float(np.sqrt(np.mean(samples ** 2)))
    print(f"   RMS (4x gain): {rms_gained:.6f}")
    print(f"   VAD threshold: 0.002000")
    
    if rms_gained < 0.002:
        print("⚠️  WARNING: Audio RMS below VAD threshold!")
        print("   This audio would be skipped by Buddy.")
        print("   Solutions:")
        print("   - Speak louder or move closer to mic")
        print("   - Increase MIC_GAIN in buddy_pi_main.py")
        print("   - Lower VAD_THRESHOLD in buddy_pi_main.py")
    else:
        print("✅ Audio RMS above threshold - would be sent to STT")
    
    # Step 3: Resample
    print(f"\n3️⃣ Resampling {MIC_RATE}Hz → {TARGET_RATE}Hz...")
    audio_16k = resample_poly(samples, TARGET_RATE, MIC_RATE).astype(np.float32)
    print(f"✅ Resampled: {len(audio_16k)} samples = {len(audio_16k)/TARGET_RATE:.1f}s")
    
    # Step 4: Send to STT server
    print(f"\n4️⃣ Connecting to STT server {STT_SERVER_IP}:{STT_PORT}...")
    try:
        uri = f"ws://{STT_SERVER_IP}:{STT_PORT}"
        async with websockets.connect(uri, timeout=10) as websocket:
            print("✅ Connected")
            
            print("5️⃣ Sending audio to Whisper...")
            await websocket.send(audio_16k.tobytes())
            
            print("6️⃣ Waiting for transcription...")
            start = time.time()
            text = await websocket.recv()
            elapsed = time.time() - start
            
            print(f"\n{'=' * 60}")
            print(f"RESULT ({elapsed:.2f}s):")
            print(f"{'=' * 60}")
            if text.strip():
                print(f"✅ '{text}'")
            else:
                print("🔇 (empty - server detected silence)")
            print(f"{'=' * 60}")
            
    except Exception as e:
        print(f"❌ STT server error: {e}")
        print("\nTroubleshooting:")
        print(f"  1. Is the server running? python3 stt_server.py")
        print(f"  2. Is the IP correct? Currently: {STT_SERVER_IP}")
        print(f"  3. Can you ping it? ping {STT_SERVER_IP}")

if __name__ == "__main__":
    print("\n🎤 SPEAK NOW after recording starts...\n")
    asyncio.run(test_stt())
