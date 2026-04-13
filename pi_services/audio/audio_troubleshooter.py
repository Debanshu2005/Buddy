#!/usr/bin/env python3
"""
INMP441 Audio Troubleshooting Script
Diagnose and fix common audio issues on Raspberry Pi 4B
"""
import subprocess
import os
import sounddevice as sd
import numpy as np
import time

class AudioTroubleshooter:
    def __init__(self):
        self.target_device = "hw:3,0"
        self.sample_rate = 48000
        self.channels = 2
        
    def run_full_troubleshooting(self):
        """Run complete troubleshooting sequence"""
        print("🔧 INMP441 TROUBLESHOOTING")
        print("=" * 40)
        
        steps = [
            ("Check ALSA Configuration", self._check_alsa_config),
            ("Verify I2S Interface", self._check_i2s_interface),
            ("Test Audio Devices", self._test_audio_devices),
            ("Check Permissions", self._check_permissions),
            ("Test Raw Capture", self._test_raw_capture),
            ("Check System Resources", self._check_system_resources),
            ("Verify Audio Routing", self._verify_audio_routing)
        ]
        
        results = {}
        for step_name, step_func in steps:
            print(f"\n🔍 {step_name}...")
            try:
                results[step_name] = step_func()
            except Exception as e:
                print(f"❌ {step_name} failed: {e}")
                results[step_name] = False
        
        self._generate_report(results)
        return results
    
    def _check_alsa_config(self):
        """Check ALSA configuration"""
        try:
            # Check if ALSA sees the device
            result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
            if result.returncode == 0:
                if "card 3" in result.stdout:
                    print("✅ ALSA detects card 3")
                    return True
                else:
                    print("❌ ALSA does not detect card 3")
                    print("Available cards:")
                    print(result.stdout)
                    return False
            else:
                print(f"❌ ALSA check failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ ALSA check error: {e}")
            return False
    
    def _check_i2s_interface(self):
        """Check I2S interface status"""
        try:
            # Check if I2S is enabled in config
            with open('/boot/config.txt', 'r') as f:
                config = f.read()
            
            i2s_enabled = 'dtparam=i2s=on' in config
            print(f"I2S enabled in config.txt: {i2s_enabled}")
            
            # Check device tree overlays
            overlay_present = 'dtoverlay=i2s-mmap' in config
            print(f"I2S overlay present: {overlay_present}")
            
            if not i2s_enabled:
                print("⚠️ I2S not enabled. Add 'dtparam=i2s=on' to /boot/config.txt")
            
            return i2s_enabled
            
        except Exception as e:
            print(f"❌ I2S check error: {e}")
            return False
    
    def _test_audio_devices(self):
        """Test audio device availability"""
        try:
            devices = sd.query_devices()\n            print(f\"Total devices found: {len(devices)}\")\n            \n            target_found = False\n            for i, device in enumerate(devices):\n                device_name = device.get('name', 'Unknown')\n                max_inputs = device.get('max_input_channels', 0)\n                \n                if \"hw:3,0\" in str(device) or device_name == 'hw:3,0':\n                    target_found = True\n                    print(f\"✅ Found target device: {device}\")\n                elif max_inputs > 0:\n                    print(f\"📱 Input device {i}: {device_name} ({max_inputs} channels)\")\n            \n            if not target_found:\n                print(\"❌ Target device hw:3,0 not found\")\n                print(\"\\nTrying alternative device names...\")\n                \n                # Try common alternatives\n                alternatives = ['hw:1,0', 'hw:2,0', 'plughw:3,0']\n                for alt in alternatives:\n                    try:\n                        test_audio = sd.rec(\n                            int(0.5 * self.sample_rate),\n                            samplerate=self.sample_rate,\n                            channels=self.channels,\n                            dtype=\"int16\",\n                            device=alt\n                        )\n                        sd.wait()\n                        print(f\"✅ Alternative device {alt} works\")\n                        return True\n                    except:\n                        print(f\"❌ Alternative device {alt} failed\")\n            \n            return target_found\n            \n        except Exception as e:\n            print(f\"❌ Device test error: {e}\")\n            return False\n    \n    def _check_permissions(self):\n        \"\"\"Check audio permissions\"\"\"\n        try:\n            # Check if user is in audio group\n            result = subprocess.run(['groups'], capture_output=True, text=True)\n            if result.returncode == 0:\n                groups = result.stdout.strip()\n                audio_group = 'audio' in groups\n                print(f\"User in audio group: {audio_group}\")\n                \n                if not audio_group:\n                    print(\"⚠️ Add user to audio group: sudo usermod -a -G audio $USER\")\n                \n                return audio_group\n            else:\n                print(f\"❌ Groups check failed: {result.stderr}\")\n                return False\n                \n        except Exception as e:\n            print(f\"❌ Permissions check error: {e}\")\n            return False\n    \n    def _test_raw_capture(self):\n        \"\"\"Test raw audio capture\"\"\"\n        try:\n            print(\"Testing 2-second capture...\")\n            \n            # Try capture with target device\n            audio = sd.rec(\n                int(2.0 * self.sample_rate),\n                samplerate=self.sample_rate,\n                channels=self.channels,\n                dtype=\"int16\",\n                device=self.target_device\n            )\n            sd.wait()\n            \n            # Analyze capture\n            left_max = np.max(np.abs(audio[:, 0]))\n            right_max = np.max(np.abs(audio[:, 1]))\n            left_rms = np.sqrt(np.mean(audio[:, 0].astype(np.float32) ** 2))\n            right_rms = np.sqrt(np.mean(audio[:, 1].astype(np.float32) ** 2))\n            \n            print(f\"Left channel  - Max: {left_max:6d}, RMS: {left_rms:8.2f}\")\n            print(f\"Right channel - Max: {right_max:6d}, RMS: {right_rms:8.2f}\")\n            \n            # Check for issues\n            if left_max == 0 and right_max == 0:\n                print(\"❌ Complete silence - check connections\")\n                return False\n            elif left_max == 0 or right_max == 0:\n                print(\"⚠️ One channel silent - check wiring\")\n                return True\n            else:\n                print(\"✅ Both channels active\")\n                return True\n                \n        except Exception as e:\n            print(f\"❌ Raw capture test failed: {e}\")\n            return False\n    \n    def _check_system_resources(self):\n        \"\"\"Check system resources\"\"\"\n        try:\n            # Check CPU usage\n            result = subprocess.run(['top', '-bn1'], capture_output=True, text=True)\n            if result.returncode == 0:\n                lines = result.stdout.split('\\n')\n                for line in lines:\n                    if 'Cpu(s):' in line or '%Cpu(s):' in line:\n                        print(f\"CPU usage: {line.strip()}\")\n                        break\n            \n            # Check memory\n            result = subprocess.run(['free', '-h'], capture_output=True, text=True)\n            if result.returncode == 0:\n                lines = result.stdout.split('\\n')\n                for line in lines:\n                    if 'Mem:' in line:\n                        print(f\"Memory: {line.strip()}\")\n                        break\n            \n            # Check for audio processes\n            result = subprocess.run(['pgrep', '-f', 'pulse|alsa'], capture_output=True, text=True)\n            if result.returncode == 0:\n                processes = result.stdout.strip().split('\\n')\n                print(f\"Audio processes running: {len(processes)}\")\n            \n            return True\n            \n        except Exception as e:\n            print(f\"❌ System resources check error: {e}\")\n            return False\n    \n    def _verify_audio_routing(self):\n        \"\"\"Verify audio routing configuration\"\"\"\n        try:\n            # Check ALSA mixer settings\n            result = subprocess.run(['amixer', 'scontrols'], capture_output=True, text=True)\n            if result.returncode == 0:\n                controls = result.stdout.count('Simple mixer control')\n                print(f\"ALSA mixer controls: {controls}\")\n            \n            # Check for PulseAudio interference\n            result = subprocess.run(['pgrep', 'pulseaudio'], capture_output=True, text=True)\n            if result.returncode == 0:\n                print(\"⚠️ PulseAudio running - may interfere with direct ALSA access\")\n                print(\"Consider: pulseaudio --kill\")\n            else:\n                print(\"✅ PulseAudio not running\")\n            \n            return True\n            \n        except Exception as e:\n            print(f\"❌ Audio routing check error: {e}\")\n            return False\n    \n    def _generate_report(self, results):\n        \"\"\"Generate troubleshooting report\"\"\"\n        print(\"\\n\" + \"=\" * 40)\n        print(\"🔧 TROUBLESHOOTING REPORT\")\n        print(\"=\" * 40)\n        \n        passed = sum(1 for result in results.values() if result)\n        total = len(results)\n        \n        print(f\"Tests passed: {passed}/{total}\")\n        print(\"\\nDetailed results:\")\n        \n        for test, result in results.items():\n            status = \"✅ PASS\" if result else \"❌ FAIL\"\n            print(f\"  {test}: {status}\")\n        \n        # Recommendations\n        print(\"\\n🔧 RECOMMENDATIONS:\")\n        \n        if not results.get(\"Check I2S Interface\", True):\n            print(\"  • Enable I2S: Add 'dtparam=i2s=on' to /boot/config.txt and reboot\")\n        \n        if not results.get(\"Check Permissions\", True):\n            print(\"  • Fix permissions: sudo usermod -a -G audio $USER\")\n        \n        if not results.get(\"Test Audio Devices\", True):\n            print(\"  • Check INMP441 wiring and connections\")\n            print(\"  • Verify I2S pins: BCK=18, WS=19, SD=20, GND=GND, VDD=3.3V\")\n        \n        if not results.get(\"Test Raw Capture\", True):\n            print(\"  • Check microphone connections\")\n            print(\"  • Verify power supply (3.3V)\")\n            print(\"  • Test with arecord: arecord -D hw:3,0 -f S16_LE -r 48000 -c 2 test.wav\")\n        \n        print(\"\\n\" + \"=\" * 40)\n\ndef main():\n    \"\"\"Main troubleshooting function\"\"\"\n    troubleshooter = AudioTroubleshooter()\n    \n    print(\"🎤 INMP441 Audio Troubleshooter\")\n    print(\"This will diagnose common audio issues\")\n    print(\"\\nPress Enter to start...\")\n    input()\n    \n    try:\n        results = troubleshooter.run_full_troubleshooting()\n        \n        # Offer to run continuous test\n        if any(results.values()):\n            print(\"\\n🔄 Run continuous audio test? (y/n): \", end=\"\")\n            if input().lower().startswith('y'):\n                print(\"\\nRunning 30-second continuous test...\")\n                print(\"Make noise to test microphone response\")\n                \n                for i in range(30):\n                    try:\n                        audio = sd.rec(\n                            int(1.0 * 48000),\n                            samplerate=48000,\n                            channels=2,\n                            dtype=\"int16\",\n                            device=\"hw:3,0\"\n                        )\n                        sd.wait()\n                        \n                        max_level = np.max(np.abs(audio))\n                        print(f\"[{i+1:2d}/30] Audio level: {max_level:6d}\", end=\"\\r\")\n                        \n                    except Exception as e:\n                        print(f\"\\n❌ Continuous test error: {e}\")\n                        break\n                \n                print(\"\\n✅ Continuous test completed\")\n        \n    except KeyboardInterrupt:\n        print(\"\\n⚠️ Troubleshooting interrupted\")\n    except Exception as e:\n        print(f\"\\n❌ Troubleshooting error: {e}\")\n\nif __name__ == \"__main__\":\n    main()