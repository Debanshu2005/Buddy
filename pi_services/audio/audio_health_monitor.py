"""
Audio Health Monitor
Continuous monitoring of INMP441 microphone health and audio system status
"""
import sounddevice as sd
import numpy as np
import time
import threading
import json
from datetime import datetime

class AudioHealthMonitor:
    def __init__(self, device=0, sample_rate=48000, channels=2):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.monitoring = False
        self.health_status = {
            'device_available': False,
            'audio_capture_working': False,
            'last_audio_level': 0,
            'last_check_time': None,
            'consecutive_failures': 0,
            'total_checks': 0,
            'success_rate': 0.0
        }
        
    def start_monitoring(self, check_interval=10):
        """Start continuous health monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(check_interval,), 
            daemon=True
        )
        self.monitor_thread.start()
        print(f"🔊 Audio health monitoring started (interval: {check_interval}s)")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
        print("🔇 Audio health monitoring stopped")
    
    def _monitor_loop(self, check_interval):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._perform_health_check()
                time.sleep(check_interval)
            except Exception as e:
                print(f"❌ Health monitor error: {e}")
                time.sleep(check_interval)
    
    def _perform_health_check(self):
        """Perform comprehensive health check"""
        self.health_status['total_checks'] += 1
        check_time = datetime.now()
        
        try:
            # Test 1: Device availability
            devices = sd.query_devices()
            device_found = any("hw:3,0" in str(d) or d.get('name') == 'hw:3,0' for d in devices)
            self.health_status['device_available'] = device_found
            
            if not device_found:
                self._log_failure("Device not found")
                return
            
            # Test 2: Audio capture
            audio = sd.rec(
                int(0.5 * self.sample_rate),  # 0.5 second test
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                device=self.device
            )\n            sd.wait()\n            \n            # Analyze audio\n            max_level = np.max(np.abs(audio))\n            self.health_status['last_audio_level'] = int(max_level)\n            self.health_status['audio_capture_working'] = max_level > 0\n            self.health_status['last_check_time'] = check_time.strftime(\"%H:%M:%S\")\n            \n            if max_level > 0:\n                self.health_status['consecutive_failures'] = 0\n                success_count = self.health_status['total_checks'] - self.health_status.get('total_failures', 0)\n                self.health_status['success_rate'] = (success_count / self.health_status['total_checks']) * 100\n                \n                # Only log if there's significant audio activity\n                if max_level > 100:\n                    print(f\"🎤 [{check_time.strftime('%H:%M:%S')}] Audio OK - Level: {max_level}\")\n            else:\n                self._log_failure(\"No audio signal\")\n                \n        except Exception as e:\n            self._log_failure(f\"Capture error: {e}\")\n    \n    def _log_failure(self, reason):\n        \"\"\"Log health check failure\"\"\"\n        self.health_status['consecutive_failures'] += 1\n        self.health_status['audio_capture_working'] = False\n        \n        if not hasattr(self.health_status, 'total_failures'):\n            self.health_status['total_failures'] = 0\n        self.health_status['total_failures'] += 1\n        \n        # Calculate success rate\n        success_count = self.health_status['total_checks'] - self.health_status['total_failures']\n        self.health_status['success_rate'] = (success_count / self.health_status['total_checks']) * 100\n        \n        timestamp = datetime.now().strftime(\"%H:%M:%S\")\n        print(f\"❌ [{timestamp}] Audio health check failed: {reason}\")\n        \n        # Alert on consecutive failures\n        if self.health_status['consecutive_failures'] >= 3:\n            print(f\"🚨 ALERT: {self.health_status['consecutive_failures']} consecutive audio failures!\")\n    \n    def get_health_report(self):\n        \"\"\"Get detailed health report\"\"\"\n        return {\n            'status': 'HEALTHY' if self.health_status['audio_capture_working'] else 'UNHEALTHY',\n            'device_available': self.health_status['device_available'],\n            'audio_working': self.health_status['audio_capture_working'],\n            'last_level': self.health_status['last_audio_level'],\n            'last_check': self.health_status['last_check_time'],\n            'consecutive_failures': self.health_status['consecutive_failures'],\n            'total_checks': self.health_status['total_checks'],\n            'success_rate': f\"{self.health_status['success_rate']:.1f}%\"\n        }\n    \n    def print_status(self):\n        \"\"\"Print current health status\"\"\"\n        report = self.get_health_report()\n        print(\"\\n🔊 AUDIO HEALTH STATUS\")\n        print(\"=\" * 30)\n        print(f\"Status: {report['status']}\")\n        print(f\"Device Available: {report['device_available']}\")\n        print(f\"Audio Working: {report['audio_working']}\")\n        print(f\"Last Audio Level: {report['last_level']}\")\n        print(f\"Last Check: {report['last_check']}\")\n        print(f\"Consecutive Failures: {report['consecutive_failures']}\")\n        print(f\"Total Checks: {report['total_checks']}\")\n        print(f\"Success Rate: {report['success_rate']}\")\n        print(\"=\" * 30)\n\ndef main():\n    \"\"\"Standalone health monitor\"\"\"\n    monitor = AudioHealthMonitor()\n    \n    try:\n        print(\"🎤 Starting Audio Health Monitor\")\n        print(\"Press Ctrl+C to stop\")\n        \n        monitor.start_monitoring(check_interval=5)\n        \n        # Interactive commands\n        while True:\n            try:\n                cmd = input(\"\\nCommands: 'status', 'report', 'quit': \").strip().lower()\n                \n                if cmd == 'status':\n                    monitor.print_status()\n                elif cmd == 'report':\n                    report = monitor.get_health_report()\n                    print(json.dumps(report, indent=2))\n                elif cmd == 'quit':\n                    break\n                else:\n                    print(\"Unknown command\")\n                    \n            except EOFError:\n                break\n                \n    except KeyboardInterrupt:\n        print(\"\\n⚠️ Interrupted\")\n    finally:\n        monitor.stop_monitoring()\n        print(\"👋 Audio health monitor stopped\")\n\nif __name__ == \"__main__\":\n    main()