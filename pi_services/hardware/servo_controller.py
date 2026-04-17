"""
ServoController — GPIO PWM servo control for camera tilt.

Servo specs:
  - Standard hobby servo (SG90 or similar)
  - Connected to GPIO 18 (physical pin 12)
  - 50 Hz PWM, duty cycle 2.5% (0°) to 12.5% (180°)
  
Positions:
  - DOWN (looking at chest/table level): ~30°
  - CENTER (eye level): ~90°
  - UP (looking at face): ~150°
"""

import RPi.GPIO as GPIO
import time
import threading

SERVO_PIN = 18  # GPIO18 (physical pin 12) — hardware PWM
PWM_FREQ  = 50 # Hz (standard servo frequency)

# Angle → duty cycle conversion
# Standard servo: 0° = 2.5% duty, 180° = 12.5% duty
def _angle_to_duty(angle: float) -> float:
    """Convert angle (0-180) to PWM duty cycle (0-100)."""
    return 2.5 + (angle / 180.0) * 10.0


class ServoController:
    """Thread-safe servo controller with smooth movement."""
    
    # Preset positions (angles in degrees)
    POS_DOWN   = 30   # Looking down (no face detected)
    POS_CENTER = 90   # Eye level (neutral)
    POS_UP     = 150  # Looking up (searching for face)
    
    def __init__(self):
        self._current_angle = self.POS_CENTER
        self._target_angle  = self.POS_CENTER
        self._moving        = False
        self._pwm           = None
        self._lock          = threading.Lock()
        
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(SERVO_PIN, GPIO.OUT)
            self._pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
            self._pwm.start(0)  # start with 0% duty (servo off)
            
            # Move to center position on startup
            self._set_angle_immediate(self.POS_CENTER)
            time.sleep(0.5)  # let servo reach position
            self._pwm.ChangeDutyCycle(0)  # turn off to prevent jitter
            
            print(f"✅ ServoController initialized on GPIO {SERVO_PIN}")
        except Exception as e:
            print(f"⚠️ ServoController init error: {e}")
            self._pwm = None
    
    def move_to(self, angle: float, smooth: bool = True):
        """
        Move servo to target angle.
        If smooth=True, moves gradually (slower but quieter).
        If smooth=False, jumps directly (faster but may jerk).
        """
        if not self._pwm:
            return
        
        angle = max(0, min(180, angle))  # clamp to 0-180
        
        with self._lock:
            self._target_angle = angle
            
            if smooth:
                threading.Thread(target=self._smooth_move, daemon=True).start()
            else:
                self._set_angle_immediate(angle)
    
    def look_up(self, smooth: bool = True):
        """Move to UP position (searching for face)."""
        print(f"👆 Servo: looking UP ({self.POS_UP}°)")
        self.move_to(self.POS_UP, smooth)
    
    def look_down(self, smooth: bool = True):
        """Move to DOWN position (face detected, lowering gaze)."""
        print(f"👇 Servo: looking DOWN ({self.POS_DOWN}°)")
        self.move_to(self.POS_DOWN, smooth)
    
    def look_center(self, smooth: bool = True):
        """Move to CENTER position (neutral eye level)."""
        print(f"👁️ Servo: looking CENTER ({self.POS_CENTER}°)")
        self.move_to(self.POS_CENTER, smooth)
    
    def _set_angle_immediate(self, angle: float):
        """Set servo angle instantly (no smoothing)."""
        if not self._pwm:
            return
        duty = _angle_to_duty(angle)
        self._pwm.ChangeDutyCycle(duty)
        self._current_angle = angle
        time.sleep(0.3)  # give servo time to move
        self._pwm.ChangeDutyCycle(0)  # turn off PWM to prevent jitter/hum
    
    def _smooth_move(self):
        """Gradually move from current to target angle."""
        if self._moving:
            return  # already moving
        
        self._moving = True
        
        try:
            steps = 20
            delay = 0.02  # 20ms between steps
            
            start = self._current_angle
            end   = self._target_angle
            delta = (end - start) / steps
            
            for i in range(steps):
                angle = start + (delta * i)
                duty = _angle_to_duty(angle)
                self._pwm.ChangeDutyCycle(duty)
                self._current_angle = angle
                time.sleep(delay)
            
            # Final position
            self._pwm.ChangeDutyCycle(_angle_to_duty(end))
            self._current_angle = end
            time.sleep(0.1)
            self._pwm.ChangeDutyCycle(0)  # turn off
            
        finally:
            self._moving = False
    
    def get_angle(self) -> float:
        """Return current servo angle."""
        return self._current_angle
    
    def cleanup(self):
        """Stop PWM and release GPIO."""
        if self._pwm:
            self._pwm.stop()
        GPIO.cleanup(SERVO_PIN)
        print("🔌 ServoController cleaned up")
