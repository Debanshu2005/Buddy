from datetime import datetime

_seen = set()

_GROUPED_PATTERNS = ["messages", "new messages", "notifications", "chats", "others"]

# Category → (decision, note)
_CATEGORIES = {
    "messaging":    ("important", "💬 Direct message"),
    "email":        ("important", "📧 Email received"),
    "social":       ("ignore",    "🌿 Social noise"),
    "entertainment":("ignore",    "🎬 Entertainment"),
    "shopping":     ("normal",    "🛒 Shopping update"),
    "finance":      ("important", "💰 Finance alert"),
    "productivity": ("normal",    "📋 Productivity"),
    "news":         ("normal",    "📰 News update"),
    "system":       ("normal",    "⚙️ System alert"),
    }

# App keyword → category
_APP_CATEGORY_MAP = {
    # messaging
    "whatsapp": "messaging", "telegram": "messaging", "signal": "messaging",
    "messenger": "messaging", "viber": "messaging", "wechat": "messaging",
    "snapchat": "messaging", "discord": "messaging", "slack": "messaging",
    "teams": "messaging", "skype": "messaging", "imessage": "messaging",
    "sms": "messaging", "messages": "messaging",
    # email
    "gmail": "email", "outlook": "email", "yahoo mail": "email",
    "protonmail": "email", "mail": "email",
    # social
    "instagram": "social", "facebook": "social", "twitter": "social",
    "x ": "social", "linkedin": "social", "reddit": "social",
    "pinterest": "social", "tumblr": "social", "tiktok": "social",
    "threads": "social",
    # entertainment
    "youtube": "entertainment", "netflix": "entertainment", "spotify": "entertainment",
    "prime video": "entertainment", "hotstar": "entertainment", "twitch": "entertainment",
    "jio cinema": "entertainment", "zee5": "entertainment",
    # shopping
    "amazon": "shopping", "flipkart": "shopping", "myntra": "shopping",
    "meesho": "shopping", "ebay": "shopping", "swiggy": "shopping",
    "zomato": "shopping", "blinkit": "shopping", "zepto": "shopping",
    # finance
    "gpay": "finance", "phonepe": "finance", "paytm": "finance",
    "bank": "finance", "upi": "finance", "cred": "finance",
    "groww": "finance", "zerodha": "finance",
    # productivity
    "notion": "productivity", "trello": "productivity", "jira": "productivity",
    "asana": "productivity", "todoist": "productivity", "calendar": "productivity",
    "google keep": "productivity", "evernote": "productivity",
    # news
    "inshorts": "news", "google news": "news", "bbc": "news",
    "times of india": "news", "ndtv": "news", "the hindu": "news",
    # system
    "android": "system", "settings": "system", "phone": "system",
    "battery": "system", "security": "system",
    }


def classify_app(app_name: str) -> tuple[str, str]:
    a = app_name.lower()
    for keyword, category in _APP_CATEGORY_MAP.items():
        if keyword in a:
            return _CATEGORIES[category]
    return "normal", f"🔍 Unknown app: {app_name}"


def is_grouped(title: str, message: str) -> bool:
    t = title.lower()
    return any(p in t for p in _GROUPED_PATTERNS) or "\n" in message


def extract_sender_message(title: str, message: str):
    if ":" in message:
        parts = message.split(":", 1)
        sender, msg = parts[0].strip(), parts[1].strip()
        if sender and msg:
            return sender, msg
    return title.strip(), message.strip()


def normalize_notification(app_name: str, title: str, message: str) -> dict:
    sender, msg = extract_sender_message(title, message)
    return {
        "app": app_name,
        "sender": sender,
        "message": msg,
        "grouped": is_grouped(title, message),
    }


def is_duplicate(sender: str, msg: str) -> bool:
    key = f"{sender}:{msg}"
    if key in _seen:
        return True
    _seen.add(key)
    if len(_seen) > 100:
        _seen.clear()
    return False


def process_notification(app_name: str, title: str, message: str) -> dict:
    """Core logic — parse, deduplicate, and classify a notification."""
    parsed = normalize_notification(app_name, title, message)
    sender, msg, grouped = parsed["sender"], parsed["message"], parsed["grouped"]

    if grouped and not msg:
        return {"status": "ignored"}

    if is_duplicate(sender, msg):
        return {"status": "duplicate"}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    decision = "normal"
    note = "🔍 Noted"

    decision, note = classify_app(app_name)

    return {
        "status": "received",
        "decision": decision,
        "sender": sender,
        "message": msg,
        "grouped": grouped,
        "timestamp": timestamp,
        "note": note,
    }


