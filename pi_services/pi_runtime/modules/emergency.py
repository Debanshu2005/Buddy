"""Emergency mixin — emergency response, webhooks, WhatsApp alerts."""
from __future__ import annotations
import os, shlex, subprocess, threading, time
from typing import Optional
import requests

class EmergencyMixin:
    def _trigger_emergency_response(self, reason: str):
        if self._emergency_active:
            self.logger.warning("Emergency response already active. Reason: %s", reason)
            return
        self._emergency_active = True
        user_name = self.active_user or "unknown user"
        message = f"Emergency check triggered for {user_name}. Reason: {reason}"
        self.logger.error(message)
        print(f"[EMERGENCY] {message}")
        self.speak("I am contacting your emergency contacts now.")

        jpeg_bytes = None
        with self._stream_lock:
            jpeg_bytes = self._stream_frame

        self._send_alert_channels(
            label="emergency",
            severity="critical",
            reason=message,
            jpeg_bytes=jpeg_bytes,
        )

        webhook_url = os.getenv("BUDDY_EMERGENCY_WEBHOOK_URL", "").strip()
        if webhook_url:
            threading.Thread(
                target=self._send_emergency_webhook,
                args=(webhook_url, message, []),
                daemon=True,
            ).start()

        call_command = os.getenv("BUDDY_EMERGENCY_CALL_COMMAND", "").strip()
        if call_command:
            threading.Thread(
                target=self._run_emergency_call_command,
                args=(call_command, message),
                daemon=True,
            ).start()

        # Reset after 5 minutes so the robot can respond again
        def _reset():
            time.sleep(300)
            self._emergency_active = False
            print("[EMERGENCY] Emergency state reset — robot is responsive again.")
        threading.Thread(target=_reset, daemon=True).start()


    def _send_emergency_webhook(self, webhook_url: str, message: str, contacts: list[str]):
        try:
            response = requests.post(
                webhook_url,
                json={
                    "message": message,
                    "active_user": self.active_user,
                    "contacts": contacts,
                    "timestamp": time.time(),
                },
                timeout=10,
            )
            if response.status_code >= 400:
                self.logger.warning("Emergency webhook returned status %s", response.status_code)
        except Exception as exc:
            self.logger.warning("Emergency webhook failed: %s", exc)


    def _send_alert_channels(
        self,
        label: str,
        severity: str,
        reason: str,
        jpeg_bytes: Optional[bytes] = None,
    ):
        message = (
            f"Buddy alert: {label}\n"
            f"Severity: {severity}\n"
            f"Reason: {reason}\n"
            "Please call or check immediately."
        )
        self.logger.info(
            "Telegram alerts temporarily disabled. Skipping alert send for %s (%s): %s",
            label,
            severity,
            reason,
        )



    def _get_whatsapp_family_numbers(self) -> list[str]:
        raw_numbers = os.getenv("BUDDY_FAMILY_WHATSAPP_NUMBERS", "").split(",")
        return [number.strip() for number in raw_numbers if number.strip()]


    def _send_whatsapp_family_alert(self, message: str, reason: str, numbers: list[str]):
        token = os.getenv("BUDDY_WHATSAPP_TOKEN", "").strip()
        phone_number_id = os.getenv("BUDDY_WHATSAPP_PHONE_NUMBER_ID", "").strip()
        api_version = os.getenv("BUDDY_WHATSAPP_API_VERSION", "v21.0").strip() or "v21.0"

        if not token or not phone_number_id:
            self.logger.warning(
                "WhatsApp alert skipped. Set BUDDY_WHATSAPP_TOKEN and BUDDY_WHATSAPP_PHONE_NUMBER_ID."
            )
            return

        url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        user_name = self.active_user or "Buddy user"
        template_name = os.getenv("BUDDY_WHATSAPP_TEMPLATE_NAME", "").strip()
        template_language = os.getenv("BUDDY_WHATSAPP_TEMPLATE_LANGUAGE", "en_US").strip() or "en_US"

        for number in numbers:
            payload = self._build_whatsapp_alert_payload(
                number=number,
                message=message,
                user_name=user_name,
                reason=reason,
                template_name=template_name,
                template_language=template_language,
            )
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                if response.status_code >= 400:
                    self.logger.warning(
                        "WhatsApp alert failed for %s: %s",
                        number,
                        response.text,
                    )
                else:
                    print(f"[EMERGENCY] WhatsApp alert sent to {number}.")
            except Exception as exc:
                self.logger.warning("WhatsApp alert error for %s: %s", number, exc)


    def _build_whatsapp_alert_payload(
        self,
        number: str,
        message: str,
        user_name: str,
        reason: str,
        template_name: str,
        template_language: str,
    ) -> dict:
        if template_name:
            return {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": number,
                "type": "template",
                "template": {
                    "name": template_name,
                    "language": {"code": template_language},
                    "components": [
                        {
                            "type": "body",
                            "parameters": [
                                {"type": "text", "text": user_name},
                                {"type": "text", "text": reason},
                            ],
                        }
                    ],
                },
            }

        return {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": number,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": (
                    f"{message}\n"
                    "Please call or check on them immediately."
                ),
            },
        }


    def _run_emergency_call_command(self, call_command: str, message: str):
        try:
            subprocess.run(
                shlex.split(call_command),
                input=message,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except Exception as exc:
            self.logger.warning("Emergency call command failed: %s", exc)


