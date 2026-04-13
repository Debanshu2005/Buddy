#!/usr/bin/env python3
"""
Simple Audio Device Scanner
Find all available input devices and test them
"""
import sounddevice as sd
import numpy as np

def scan_devices():
    print("🔍 Scanning Audio Devices")
    print("=" * 30)
    
    devices = sd.query_devices()
    print(f"Total devices: {len(devices)}")
    
    input_devices = []
    
    for i, device in enumerate(devices):
        name = device.get('name', 'Unknown')
        inputs = device.get('max_input_channels', 0)
        outputs = device.get('max_output_channels', 0)
        
        print(f"\nDevice {i}: {name}")
        print(f"  Inputs: {inputs}, Outputs: {outputs}")
        
        if inputs > 0:
            input_devices.append(i)
            print(f"  ✅ Has input capability")
            
            # Test the device
            try:
                print(f"  Testing device {i}...")
                audio = sd.rec(
                    int(0.5 * 48000),
                    samplerate=48000,
                    channels=min(2, inputs),
                    dtype="int16",
                    device=i
                )
                sd.wait()
                
                max_level = np.max(np.abs(audio))
                print(f"  Test result: max level {max_level}")
                
                if max_level > 0:
                    print(f"  🎤 WORKING INPUT DEVICE!")
                else:
                    print(f"  📵 Silent (but functional)")
                    
            except Exception as e:
                print(f"  ❌ Test failed: {e}")
        else:
            print(f"  📤 Output only")
    
    if input_devices:
        print(f"\n🎤 Found {len(input_devices)} input device(s): {input_devices}")
        print(f"💡 Use device={input_devices[0]} in your code")
    else:
        print("\n❌ No input devices found")
        print("💡 Try plugging in a USB microphone")

if __name__ == "__main__":
    scan_devices()