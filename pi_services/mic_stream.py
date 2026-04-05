import sounddevice as sd
import numpy as np
import threading

RATE = 48000
CHANNELS = 1
BUFFER_SECONDS = 4


def find_usb_mic_device() -> int:
    """
    Auto-detect USB microphone device index via sounddevice.
    Returns the device index, or None to use the system default.
    """
    try:
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            name = dev.get("name", "").lower()
            if dev.get("max_input_channels", 0) > 0 and "usb" in name:
                print(f"🎤 USB mic found: [{i}] {dev['name']}")
                return i
        print("⚠️ No USB mic found via sounddevice, using system default")
    except Exception as e:
        print(f"⚠️ USB mic detection error: {e}")
    return None  # sounddevice will use the OS default


class MicStream:
    def __init__(self, device=None):
        """
        device: explicitly pass a device index/name, or leave None to auto-detect USB mic.
        """
        self.device = device if device is not None else find_usb_mic_device()
        self.buffer_size = RATE * BUFFER_SECONDS
        self.buffer = np.zeros(self.buffer_size, dtype=np.int16)
        self.write_pos = 0
        self.lock = threading.Lock()
        self.stream = None

    def _callback(self, indata, frames, time, status):
        if status:
            print("⚠️ mic status:", status)
        # First (and only) channel
        mono = indata[:, 0].astype(np.int16)
        with self.lock:
            n = len(mono)
            end = self.write_pos + n
            if end < self.buffer_size:
                self.buffer[self.write_pos:end] = mono
            else:
                part = self.buffer_size - self.write_pos
                self.buffer[self.write_pos:] = mono[:part]
                self.buffer[:n - part] = mono[part:]
            self.write_pos = (self.write_pos + n) % self.buffer_size

    def start(self):
        if self.stream:
            return
        self.stream = sd.InputStream(
            samplerate=RATE,
            channels=CHANNELS,
            dtype="int16",
            device=self.device,  # None = system default if USB not found
            callback=self._callback,
            blocksize=2048,
            latency="high",
        )
        self.stream.start()
        device_label = f"device [{self.device}]" if self.device is not None else "default device"
        print(f"🎤 Mic stream started ({device_label})")

    def stop(self):
        """Stop and close the mic stream cleanly."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("🎤 Mic stream stopped")

    def read(self, seconds=1.0):
        samples = int(RATE * seconds)
        with self.lock:
            if samples > self.buffer_size:
                return None
            start = (self.write_pos - samples) % self.buffer_size
            if start + samples < self.buffer_size:
                return self.buffer[start:start + samples].copy()
            else:
                return np.concatenate((
                    self.buffer[start:],
                    self.buffer[:samples - (self.buffer_size - start)]
                ))
