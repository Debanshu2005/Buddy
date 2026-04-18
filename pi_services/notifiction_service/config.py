import os

PHONE_IP   = os.getenv("BUDDY_PHONE_IP", "192.168.0.106")
PHONE_PORT = int(os.getenv("BUDDY_PHONE_PORT", "8082"))
