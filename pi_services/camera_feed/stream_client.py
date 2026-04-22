"""
Pi Camera Stream Client
Receives the MJPEG stream from the PC and provides frames to other Pi services.
Run this on the Pi: python stream_client.py
Set PC_IP to your laptop's local IP address.
"""

import cv2
import urllib.request
import numpy as np

PC_IP = os.getenv("BUDDY_PC_CAMERA_IP", "buddypc.local")
PC_PORT = 5000
STREAM_URL = f"http://{PC_IP}:{PC_PORT}/video"


def get_frame_stream():
    """Generator that yields decoded frames from the PC stream."""
    stream = urllib.request.urlopen(STREAM_URL)
    buffer = b""
    while True:
        buffer += stream.read(4096)
        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9")
        if start != -1 and end != -1:
            jpg = buffer[start:end + 2]
            buffer = buffer[end + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame


if __name__ == "__main__":
    print(f"[Camera Client] Connecting to {STREAM_URL}")
    for i, frame in enumerate(get_frame_stream()):
        if i % 30 == 0:
            print(f"[Camera Client] Receiving frames... shape={frame.shape}")
        if i >= 90:  # print for 3 seconds then exit
            print("[Camera Client] Stream working correctly.")
            break
