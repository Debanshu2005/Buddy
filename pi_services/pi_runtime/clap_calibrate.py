#!/usr/bin/env python3
"""
Clap calibration tool for Buddy.
Run this on the Pi to find the right BUDDY_CLAP_THRESHOLD value.

Usage:
    cd /home/debanshu05/projects/Pi_Buddy/pi_services
    python pi_runtime/clap_calibrate.py
"""
import subprocess
import numpy as np
import time

MIC_RATE       = 48000
CHUNK_SECS     = 0.02
BYTES_PER_CHUNK = int(MIC_RATE * CHUNK_SECS) * 2

def find_mic_device() -> str:
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if line.startswith("card"):
                card = line.split(":")[0].strip().split()[-1]
                dev  = line.split("device")[-1].strip().split(":")[0].strip()
                return f"plughw:{card},{dev}"
    except Exception:
        pass
    return "default"

def main():
    device = find_mic_device()
    print(f"Using mic: {device}")
    print("=" * 50)
    print("This tool shows live RMS values from your mic.")
    print("  - Stay quiet for 5s to see your noise floor")
    print("  - Then clap loudly a few times")
    print("  - Set BUDDY_CLAP_THRESHOLD to ~50% of your clap peak")
    print("=" * 50)
    print("Starting in 2 seconds...\n")
    time.sleep(2)

    proc = subprocess.Popen(
        ["arecord", "-D", device, "-f", "S16_LE", "-r", str(MIC_RATE), "-c", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    max_rms   = 0.0
    noise_floor = []
    start     = time.time()
    collecting_noise = True

    try:
        while True:
            raw = proc.stdout.read(BYTES_PER_CHUNK)
            if not raw or len(raw) < BYTES_PER_CHUNK:
                break

            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms   = float(np.sqrt(np.mean(chunk ** 2)))
            elapsed = time.time() - start

            if collecting_noise and elapsed < 5.0:
                noise_floor.append(rms)
                bar = int(rms * 200)
                print(f"\r[Noise floor] RMS={rms:.4f}  {'|' * bar:<20}", end="", flush=True)
            else:
                if collecting_noise:
                    collecting_noise = False
                    avg_noise = sum(noise_floor) / max(1, len(noise_floor))
                    print(f"\n\nNoise floor avg: {avg_noise:.4f}")
                    print("Now CLAP loudly several times!\n")

                if rms > max_rms:
                    max_rms = rms

                bar = int(rms * 100)
                marker = " ← CLAP!" if rms > 0.1 else ""
                print(f"\rRMS={rms:.4f}  max={max_rms:.4f}  {'|' * min(bar, 50):<50}{marker}", end="", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        proc.kill()
        proc.wait()

    avg_noise = sum(noise_floor) / max(1, len(noise_floor))
    suggested = round(max_rms * 0.5, 3)
    print(f"\n\n{'=' * 50}")
    print(f"Noise floor : {avg_noise:.4f}")
    print(f"Clap peak   : {max_rms:.4f}")
    print(f"Suggested threshold: {suggested}")
    print(f"\nSet this in your environment:")
    print(f"  export BUDDY_CLAP_THRESHOLD={suggested}")
    print("=" * 50)

if __name__ == "__main__":
    main()
