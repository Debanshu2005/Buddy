#!/usr/bin/env python3
"""
Debug INMP441 with exact buddy_pi configuration
"""
import sounddevice as sd
import numpy as np
from scipy import signal
import vosk
import json
import os

def test_buddy_pi_audio():
    """Test exact same config as buddy_pi"""
    
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i}: {device['name']} (inputs: {device['max_input_channels']})")
    
    # Test different configurations
    configs = [
        ("hw:3,0", 48000, 2, "int32"),
        ("hw:3,0", 48000, 2, "int16"),
        (3, 48000, 2, "int32"),  # Try device index instead of name
        (3, 48000, 2, "int16"),
    ]
    
    for device, rate, channels, dtype in configs:
        print(f"\nTesting: device={device}, rate={rate}, channels={channels}, dtype={dtype}")
        
        try:
            audio = sd.rec(
                int(1.0 * rate),
                samplerate=rate,
                channels=channels,
                dtype=dtype,
                device=device
            )
            sd.wait()
            
            # Check both channels
            if channels == 2:
                left_level = np.max(np.abs(audio[:, 0]))
                right_level = np.max(np.abs(audio[:, 1]))
                print(f"  Left channel: {left_level:.0f}")
                print(f"  Right channel: {right_level:.0f}")
                
                if left_level > 0 or right_level > 0:
                    print(f"  ✅ Audio detected!")
                    return device, rate, channels, dtype
            else:
                level = np.max(np.abs(audio))
                print(f"  Audio level: {level:.0f}")
                if level > 0:
                    print(f"  ✅ Audio detected!")
                    return device, rate, channels, dtype
            
            print(f"  ❌ No audio")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n❌ No working audio configuration found")
    return None, None, None, None

if __name__ == "__main__":
    print("Speak loudly during tests...")
    device, rate, channels, dtype = test_buddy_pi_audio()
    if device is not None:
        print(f"\n✅ Working config: device={device}, rate={rate}, channels={channels}, dtype={dtype}")
    else:
        print("\n❌ INMP441 not working with any configuration")