#!/usr/bin/env python3
# csi_camera_helper.py - SYSTEM PYTHON ONLY
from picamera2 import Picamera2
from time import sleep
import os

OUT = "/tmp/csi_frame.jpg"

cam = Picamera2()
cam.start()
sleep(1.5)

print("CSI camera helper started")

try:
    while True:
        cam.capture_file(OUT)
        sleep(0.05)
except KeyboardInterrupt:
    cam.stop()
    print("CSI camera helper stopped")
