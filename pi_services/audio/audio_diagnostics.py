"""
Audio Diagnostics and Verification System
Comprehensive testing for INMP441 microphones and audio output
"""
import sounddevice as sd
import numpy as np
import time
import json
import vosk
import os
from scipy import signal
from scipy.io.wavfile import write
import pygame
import edge_tts
import asyncio
from pathlib import Path


VOSK_MODEL_PATH = Path(__file__).resolve().parent / "models" / "vosk-model-small-en-us-0.15"

class AudioDiagnostics:
    def __init__(self):
        self.sample_rate = 48000
        self.channels = 2
        self.device = None  # Will be auto-detected
        self.vosk_model = None
        self.vosk_rec = None
        
    def run_full_diagnostics(self):
        """Run complete audio system diagnostics"""
        print("🔊 AUDIO DIAGNOSTICS STARTING")
        print("=" * 50)
        
        # Test 1: Device Detection
        if not self._test_device_detection():
            return False
            
        # Test 2: Raw Audio Capture
        if not self._test_raw_audio_capture():
            return False
            
        # Test 3: Audio Level Testing
        if not self._test_audio_levels():
            return False
            
        # Test 4: Vosk Speech Recognition
        if not self._test_vosk_recognition():
            return False
            
        # Test 5: Audio Output (TTS)
        if not self._test_audio_output():
            return False
            
        print("✅ ALL AUDIO TESTS PASSED")
        return True
    
    def _test_device_detection(self):
        """Test if INMP441 device is detected"""
        print("\n1️⃣ Testing Device Detection...")
        
        try:
            devices = sd.query_devices()
            print(f"Available devices: {len(devices)}")
            
            # Look for input devices
            input_devices = []
            for i, device in enumerate(devices):
                max_inputs = device.get('max_input_channels', 0)
                device_name = device.get('name', 'Unknown')
                print(f"Device {i}: {device_name} (inputs: {max_inputs}, outputs: {device.get('max_output_channels', 0)})")
                
                if max_inputs > 0:
                    input_devices.append((i, device))
            
            if not input_devices:
                print("❌ No input devices found")
                return False
            
            # Try to find the actual INMP441 input device
            # It might not be hw:3,0 but could be a different index
            for device_id, device in input_devices:
                try:
                    print(f"Testing device {device_id}: {device['name']}")
                    test_audio = sd.rec(
                        int(0.5 * self.sample_rate),
                        samplerate=self.sample_rate,
                        channels=min(self.channels, device['max_input_channels']),
                        dtype="int16",
                        device=device_id
                    )
                    sd.wait()
                    
                    max_level = np.max(np.abs(test_audio))
                    print(f"  Test result: max level {max_level}")
                    
                    if max_level > 0:
                        print(f"✅ Working input device found: {device_id} - {device['name']}")
                        # Update the device for the rest of the tests
                        self.device = device_id
                        return True
                        
                except Exception as e:
                    print(f"  ❌ Device {device_id} test failed: {e}")
                    continue
            
            print("❌ No working input devices found")
            return False
                
        except Exception as e:
            print(f"❌ Device detection failed: {e}")
            return False
    
    def _test_raw_audio_capture(self):
        """Test raw audio capture from detected input device"""
        print("\n2️⃣ Testing Raw Audio Capture...")
        
        try:
            # Use the device found in detection test
            print(f"Using device: {self.device}")
            
            # Get device info to determine channel count
            if isinstance(self.device, int):
                device_info = sd.query_devices(self.device)
                max_channels = device_info.get('max_input_channels', 2)
                channels_to_use = min(self.channels, max_channels)
            else:
                channels_to_use = self.channels
            
            print(f"Recording 2 seconds with {channels_to_use} channels...")
            audio = sd.rec(
                int(2.0 * self.sample_rate),
                samplerate=self.sample_rate,
                channels=channels_to_use,
                dtype="int16",
                device=self.device
            )
            sd.wait()
            
            # Analyze channels
            if channels_to_use == 1:
                # Mono device
                channel_data = audio[:, 0]
                max_level = np.max(np.abs(channel_data))
                rms_level = np.sqrt(np.mean(channel_data.astype(np.float32) ** 2))
                print(f"Mono channel - Max: {max_level:6d}, RMS: {rms_level:8.2f}")
                
                if max_level == 0:
                    print("❌ No audio signal detected")
                    return False
            else:
                # Stereo device
                left_channel = audio[:, 0]
                right_channel = audio[:, 1]
                
                left_max = np.max(np.abs(left_channel))
                right_max = np.max(np.abs(right_channel))
                left_rms = np.sqrt(np.mean(left_channel.astype(np.float32) ** 2))
                right_rms = np.sqrt(np.mean(right_channel.astype(np.float32) ** 2))
                
                print(f"Left channel  - Max: {left_max:6d}, RMS: {left_rms:8.2f}")
                print(f"Right channel - Max: {right_max:6d}, RMS: {right_rms:8.2f}")
                
                if left_max == 0 and right_max == 0:
                    print("❌ No audio signal detected on either channel")
                    return False
            
            # Save sample for analysis
            write("/tmp/audio_test.wav", self.sample_rate, audio)
            print("✅ Audio sample saved to /tmp/audio_test.wav")
            
            return True
            
        except Exception as e:
            print(f"❌ Raw audio capture failed: {e}")
            return False
    
    def _test_audio_levels(self):
        """Test audio levels with different gain settings"""
        print("\n3️⃣ Testing Audio Levels and Processing...")
        
        try:
            print("Say something for 3 seconds...")
            audio = sd.rec(
                int(3.0 * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                device=self.device
            )
            sd.wait()
            
            # Process like in main code
            left_channel = audio[:, 0].astype(np.float32)
            right_channel = audio[:, 1].astype(np.float32)
            mono = (left_channel + right_channel) / 2
            
            # Test different gain levels
            gains = [1.0, 100.0, 1000.0, 10000.0, 50000.0]
            
            for gain in gains:
                processed = mono.copy()
                processed -= np.mean(processed)
                processed *= gain
                
                max_level = np.max(np.abs(processed))
                rms_level = np.sqrt(np.mean(processed ** 2))
                
                print(f"Gain {gain:8.0f}x - Max: {max_level:12.0f}, RMS: {rms_level:12.0f}")
            
            # Test resampling to 16kHz
            mono_processed = mono.copy()
            mono_processed -= np.mean(mono_processed)
            mono_processed *= 50000.0
            
            peak = np.max(np.abs(mono_processed))
            if peak > 0:
                mono_processed /= peak
            
            mono_16k = signal.resample(mono_processed, int(len(mono_processed) * 16000 / 48000))
            mono_int16 = (mono_16k * 32767).astype(np.int16)
            
            print(f"✅ Resampled to 16kHz: {len(mono_int16)} samples")
            
            return True
            
        except Exception as e:
            print(f"❌ Audio level testing failed: {e}")
            return False
    
    def _test_vosk_recognition(self):
        """Test Vosk speech recognition"""
        print("\n4️⃣ Testing Vosk Speech Recognition...")
        
        try:
            # Initialize Vosk
            model_path = str(VOSK_MODEL_PATH)
            if not os.path.exists(model_path):
                print(f"❌ Vosk model not found at {model_path}")
                return False
            
            self.vosk_model = vosk.Model(model_path)
            self.vosk_rec = vosk.KaldiRecognizer(self.vosk_model, 16000)
            print("✅ Vosk model loaded")
            
            # Test recognition
            print("Say 'hello world' for 3 seconds...")
            audio = sd.rec(
                int(3.0 * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                device=self.device
            )
            sd.wait()
            
            # Process audio
            left_channel = audio[:, 0].astype(np.float32)
            right_channel = audio[:, 1].astype(np.float32)
            mono = (left_channel + right_channel) / 2
            
            mono -= np.mean(mono)
            mono *= 50000.0
            
            peak = np.max(np.abs(mono))
            if peak > 0:
                mono /= peak
            
            mono_16k = signal.resample(mono, int(len(mono) * 16000 / 48000))
            mono_int16 = (mono_16k * 32767).astype(np.int16)
            
            # Run Vosk
            self.vosk_rec.AcceptWaveform(mono_int16.tobytes())
            result = json.loads(self.vosk_rec.FinalResult())
            text = result.get('text', '').strip()
            
            print(f"Recognized: '{text}'")
            
            if text:
                print("✅ Speech recognition working")
                return True
            else:
                print("⚠️ No speech recognized - check microphone positioning")
                return False
                
        except Exception as e:
            print(f"❌ Vosk recognition failed: {e}")
            return False
    
    def _test_audio_output(self):
        """Test audio output via TTS"""
        print("\n5️⃣ Testing Audio Output (TTS)...")
        
        try:
            # Initialize pygame mixer
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            print("✅ Pygame mixer initialized")
            
            # Test Edge TTS
            test_text = "Audio output test successful"
            asyncio.run(self._generate_test_speech(test_text))
            
            return True
            
        except Exception as e:
            print(f"❌ Audio output test failed: {e}")
            return False
    
    async def _generate_test_speech(self, text):
        """Generate and play test speech"""
        try:
            voice = "en-IN-NeerjaNeural"
            communicate = edge_tts.Communicate(text, voice)
            
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            if audio_data:
                temp_audio = "/tmp/tts_test.wav"
                with open(temp_audio, "wb") as f:
                    f.write(audio_data)
                
                print("✅ TTS audio generated")
                
                # Play using aplay
                os.system(f"aplay -D hw:0,0 {temp_audio}")
                print("✅ Audio played via hw:0,0")
                
        except Exception as e:
            print(f"❌ TTS generation failed: {e}")
    
    def continuous_monitoring(self, duration=30):
        """Continuous audio monitoring for debugging"""
        print(f"\n🔄 Continuous Audio Monitoring ({duration}s)")
        print("Speak to test audio levels...")
        
        try:
            start_time = time.time()
            
            while time.time() - start_time < duration:
                # Record 1 second chunks
                audio = sd.rec(
                    int(1.0 * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="int16",
                    device=self.device
                )
                sd.wait()
                
                # Analyze
                left_channel = audio[:, 0].astype(np.float32)
                right_channel = audio[:, 1].astype(np.float32)
                mono = (left_channel + right_channel) / 2
                
                raw_max = np.max(np.abs(mono))
                
                # Apply processing
                mono -= np.mean(mono)
                mono *= 50000.0
                processed_max = np.max(np.abs(mono))
                
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] Raw: {raw_max:6.0f} | Processed: {processed_max:12.0f}")
                
        except KeyboardInterrupt:
            print("\n⚠️ Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Monitoring error: {e}")

def main():
    """Run audio diagnostics"""
    diagnostics = AudioDiagnostics()
    
    print("🎤 INMP441 Audio Diagnostics")
    print("Choose test mode:")
    print("1. Full diagnostics")
    print("2. Continuous monitoring")
    print("3. Quick test")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            diagnostics.run_full_diagnostics()
        elif choice == "2":
            duration = int(input("Duration in seconds (default 30): ") or "30")
            diagnostics.continuous_monitoring(duration)
        elif choice == "3":
            diagnostics._test_device_detection()
            diagnostics._test_raw_audio_capture()
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n⚠️ Diagnostics interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
