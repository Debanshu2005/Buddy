from fastapi import FastAPI, Request
from datetime import datetime
import uvicorn

app = FastAPI()

# 🧠 Memory to avoid duplicates
last_notifications = set()


# 🧠 Detect grouped notifications (generic)
def is_grouped(title: str, message: str):
    t = title.lower()
    m = message.lower()

    patterns = [
        "messages",
        "new messages",
        "notifications",
        "chats",
        "others",
    ]

    return any(p in t for p in patterns) or "\n" in message


# 🧠 Try extracting sender/message from messy text
def extract_sender_message(title: str, message: str):
    # Case: "Riya: call me"
    if ":" in message:
        parts = message.split(":", 1)
        sender = parts[0].strip()
        msg = parts[1].strip()

        if sender and msg:
            return sender, msg

    # fallback
    return title.strip(), message.strip()


# 🧠 Clean + normalize notification
def normalize_notification(app_name, title, message):
    grouped = is_grouped(title, message)

    sender, msg = extract_sender_message(title, message)

    return {
        "app": app_name,
        "sender": sender,
        "message": msg,
        "grouped": grouped,
    }


# 🧠 Deduplication
def is_duplicate(sender, msg):
    key = f"{sender}:{msg}"

    if key in last_notifications:
        return True

    last_notifications.add(key)

    # prevent memory overflow
    if len(last_notifications) > 100:
        last_notifications.clear()

    return False


@app.post("/notify")
async def receive_notification(req: Request):
    data = await req.json()

    app_name = data.get("app", "Unknown")
    title = data.get("title", "")
    message = data.get("message", "")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 🧠 Normalize
    parsed = normalize_notification(app_name, title, message)

    sender = parsed["sender"]
    msg = parsed["message"]
    grouped = parsed["grouped"]

    # ❌ Ignore bad grouped notifications
    if grouped and not msg:
        print("🚫 Ignored empty grouped notification")
        return {"status": "ignored"}

    # ❌ Ignore duplicates
    if is_duplicate(sender, msg):
        print("🔁 Duplicate ignored")
        return {"status": "duplicate"}

    print("\n📩 NEW NOTIFICATION")
    print(f"🕒 {timestamp}")
    print(f"📱 App: {app_name}")
    print(f"👤 Sender: {sender}")
    print(f"💬 Message: {msg}")

    # 🧠 Intelligence layer
    decision = "normal"

    app_lower = app_name.lower()

    if "whatsapp" in app_lower:
        decision = "important"
        print("⚡ BUDDY: Message detected (WhatsApp)")

    elif "instagram" in app_lower or "facebook" in app_lower:
        decision = "ignore"
        print("🌿 BUDDY: Social noise")

    else:
        print("🔍 BUDDY: Noted")

    if grouped:
        print("⚠️ Note: Derived from grouped notification")

    return {
        "status": "received",
        "decision": decision,
        "sender": sender,
        "message": msg
    }


@app.get("/")
def home():
    return {"message": "BUDDY phone link active"}


if __name__ == "__main__":
    uvicorn.run(
        "phone:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )