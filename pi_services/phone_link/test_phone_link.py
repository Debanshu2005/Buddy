"""
Demo: two ways to use the phone_link package

1. Direct function call  — no server needed, great for testing/integration
2. FastAPI app           — mount the router and run a server
"""

# ── 1. Direct usage ──────────────────────────────────────────────────────────
from phone_link import process_notification

samples = [
    ("WhatsApp", "Riya", "Riya: call me"),
    ("Instagram", "Likes", "3 people liked your photo"),
    ("WhatsApp", "Riya", "Riya: call me"),          # duplicate → ignored
    ("Gmail", "New messages", "5 new messages\nfrom team"),  # grouped
    ("Telegram", "Alex", "Alex: hey, you there?"),
]

print("── Direct function demo ──")
for app, title, msg in samples:
    result = process_notification(app, title, msg)
    print(f"[{app}] → {result}")


# ── 2. FastAPI server usage ───────────────────────────────────────────────────
import uvicorn
from fastapi import FastAPI
from phone_link import router

app = FastAPI()
app.include_router(router)          # mounts /notify and /

if __name__ == "__main__":
    # Run: python test_phone_link.py
    # Then POST to http://localhost:8000/notify with JSON body:
    # {"app": "WhatsApp", "title": "Riya", "message": "Riya: call me"}
    uvicorn.run(app, host="0.0.0.0", port=8000)
