import logging
import os

import requests

logger = logging.getLogger(__name__)

PHONE_IP   = os.getenv("BUDDY_PHONE_IP", "192.168.1.5")
PHONE_PORT = int(os.getenv("BUDDY_PHONE_PORT", "8080"))


def trigger_emergency_sms(context: str) -> bool:
    """Send an emergency SMS via Automate HTTP server on the phone.

    Returns True if the request was delivered, False otherwise.
    """
    url = f"http://{PHONE_IP}:{PHONE_PORT}/emergency"
    msg = f"BUDDY ALERT: {context}"
    try:
        requests.get(url, params={"msg": msg}, timeout=5)
        logger.info("Emergency SMS triggered: %s", msg)
        return True
    except requests.exceptions.ConnectionError:
        logger.warning("SMS trigger failed — phone unreachable at %s:%s", PHONE_IP, PHONE_PORT)
    except requests.exceptions.Timeout:
        logger.warning("SMS trigger timed out")
    except Exception as e:
        logger.warning("SMS trigger error: %s", e)
    return False
