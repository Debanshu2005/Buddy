"""Telegram alert sender + Imgur for image hosting."""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Optional

import requests

LOGGER = logging.getLogger(__name__)

_last_alert_time: dict[str, float] = {}
_COOLDOWN_SECONDS = 60


def _upload_to_imgur(jpeg_bytes: bytes) -> Optional[str]:
    client_id = os.getenv("IMGUR_CLIENT_ID", "")
    if not client_id:
        LOGGER.warning("[Alert] IMGUR_CLIENT_ID not set — skipping image upload")
        return None
    try:
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        resp = requests.post(
            "https://api.imgur.com/3/image",
            headers={"Authorization": f"Client-ID {client_id}"},
            data={"image": b64, "type": "base64"},
            timeout=15,
        )
        if resp.status_code == 200:
            url = resp.json()["data"]["link"]
            print(f"[Alert] Image uploaded to Imgur: {url}")
            return url
        LOGGER.warning("[Alert] Imgur upload failed: %s", resp.text[:200])
    except Exception as exc:
        LOGGER.warning("[Alert] Imgur error: %s", exc)
    return None


def send_whatsapp_alert(
    label: str,
    severity: str,
    reason: str,
    jpeg_bytes: Optional[bytes] = None,
    phone: Optional[str] = None,
    apikey: Optional[str] = None,
) -> bool:
    """Send alert via Telegram bot with optional photo. Returns True if sent."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        LOGGER.warning("[Alert] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — alert skipped")
        return False

    now = time.time()
    if now - _last_alert_time.get(label, 0) < _COOLDOWN_SECONDS:
        LOGGER.info("[Alert] Cooldown active for '%s' — skipped", label)
        return False
    _last_alert_time[label] = now

    emoji = {"critical": "🚨", "high": "⚠️", "medium": "🔔"}.get(severity, "ℹ️")
    message = (
        f"{emoji} *BUDDY ALERT*\n"
        f"Event: `{label}`\n"
        f"Severity: *{severity.upper()}*\n"
        f"Reason: {reason}"
    )

    print(f"[Alert] Sending Telegram alert: {label} ({severity})")

    # try sending photo directly
    if jpeg_bytes:
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendPhoto",
                data={"chat_id": chat_id, "caption": message, "parse_mode": "Markdown"},
                files={"photo": ("snapshot.jpg", jpeg_bytes, "image/jpeg")},
                timeout=15,
            )
            if resp.status_code == 200:
                print(f"[Alert] ✅ Telegram photo sent: {label}")
                return True
            LOGGER.warning("[Alert] Telegram sendPhoto failed: %s — falling back to text", resp.text[:200])
        except Exception as exc:
            LOGGER.warning("[Alert] Telegram sendPhoto error: %s", exc)

    # text only fallback
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=15,
        )
        if resp.status_code == 200:
            print(f"[Alert] ✅ Telegram message sent: {label}")
            return True
        LOGGER.warning("[Alert] Telegram sendMessage failed: %s", resp.text[:200])
    except Exception as exc:
        LOGGER.warning("[Alert] Telegram error: %s", exc)

    return False
