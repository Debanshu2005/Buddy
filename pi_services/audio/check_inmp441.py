#!/usr/bin/env python3
"""
Quick INMP441 Hardware Check
Check if INMP441 is properly configured and suggest fixes
"""
import subprocess
import os
import sounddevice as sd

def check_inmp441_setup():
    print("🔍 INMP441 Hardware Setup Check")
    print("=" * 40)
    
    # 1. Check I2S configuration
    print("\n1️⃣ Checking I2S Configuration...")
    try:
        with open('/boot/config.txt', 'r') as f:
            config = f.read()
        
        i2s_enabled = 'dtparam=i2s=on' in config
        print(f"I2S enabled: {'✅' if i2s_enabled else '❌'}")
        
        if not i2s_enabled:
            print("❌ I2S not enabled!")
            print("Fix: Add 'dtparam=i2s=on' to /boot/config.txt and reboot")
            return False
            
    except Exception as e:
        print(f"❌ Cannot read /boot/config.txt: {e}")
        return False
    
    # 2. Check device tree overlays
    print("\n2️⃣ Checking Device Tree Overlays...")
    try:
        result = subprocess.run(['ls', '/proc/device-tree/soc/'], capture_output=True, text=True)
        if 'i2s@' in result.stdout:
            print("✅ I2S device tree node found")
        else:
            print("❌ I2S device tree node not found")
            print("Check: Ensure I2S overlay is loaded")
    except Exception as e:
        print(f"⚠️ Cannot check device tree: {e}")
    
    # 3. Check ALSA cards
    print("\n3️⃣ Checking ALSA Sound Cards...")
    try:
        result = subprocess.run(['cat', '/proc/asound/cards'], capture_output=True, text=True)
        print("Available sound cards:")
        print(result.stdout)
        
        if 'googlevoicehat' in result.stdout.lower():
            print("✅ Google VoiceHAT detected (output only)")
        
        # Look for actual input cards
        lines = result.stdout.split('\n')
        input_cards = []
        for line in lines:
            if 'USB' in line or 'capture' in line.lower() or 'input' in line.lower():
                input_cards.append(line.strip())
        
        if input_cards:
            print("📱 Potential input devices:")
            for card in input_cards:
                print(f"  {card}")
        else:
            print("❌ No obvious input devices found")
            
    except Exception as e:
        print(f"❌ Cannot check ALSA cards: {e}")
    
    # 4. Test all available devices
    print("\n4️⃣ Testing All Audio Devices...")
    try:
        devices = sd.query_devices()
        working_inputs = []
        
        for i, device in enumerate(devices):
            max_inputs = device.get('max_input_channels', 0)
            device_name = device.get('name', 'Unknown')
            
            if max_inputs > 0:
                try:
                    # Quick test
                    audio = sd.rec(
                        int(0.2 * 48000),  # 0.2 second test
                        samplerate=48000,
                        channels=min(2, max_inputs),
                        dtype="int16",
                        device=i
                    )
                    sd.wait()
                    
                    max_level = np.max(np.abs(audio))
                    print(f"✅ Device {i}: {device_name} - Test level: {max_level}")
                    working_inputs.append((i, device_name, max_inputs))
                    
                except Exception as e:
                    print(f"❌ Device {i}: {device_name} - Failed: {e}")
        
        if working_inputs:
            print(f"\n🎤 Found {len(working_inputs)} working input device(s):")
            for device_id, name, channels in working_inputs:
                print(f"  Device {device_id}: {name} ({channels} channels)")
            
            # Suggest the best device
            best_device = working_inputs[0]
            print(f"\n💡 Recommended device: {best_device[0]} - {best_device[1]}")
            print(f"Update your code to use device={best_device[0]} instead of 'hw:3,0'")
            return True
        else:
            print("\n❌ No working input devices found!")
            return False
            
    except Exception as e:
        print(f"❌ Device testing failed: {e}")
        return False

def suggest_fixes():
    print("\n🔧 INMP441 Setup Suggestions:")
    print("=" * 40)
    
    print("1. Hardware Connections:")
    print("   VDD → 3.3V")
    print("   GND → Ground") 
    print("   SD  → GPIO 20 (Pin 38)")
    print("   WS  → GPIO 19 (Pin 35)")
    print("   SCK → GPIO 18 (Pin 12)")
    
    print("\n2. Software Configuration:")
    print("   Add to /boot/config.txt:")
    print("   dtparam=i2s=on")
    print("   dtoverlay=i2s-mmap")
    
    print("\n3. Create ALSA configuration:")
    print("   Create /etc/asound.conf with INMP441 setup")
    
    print("\n4. Alternative: Use USB microphone")
    print("   Plug in USB microphone for immediate testing")

def main():
    try:
        import numpy as np
        success = check_inmp441_setup()
        
        if not success:
            suggest_fixes()
        
        print(f"\n{'✅ Setup looks good!' if success else '❌ Setup needs fixes'}")
        
    except ImportError:
        print("❌ numpy not available - install with: pip install numpy")
    except Exception as e:
        print(f"❌ Check failed: {e}")

if __name__ == "__main__":
    main()