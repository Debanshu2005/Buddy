import logging

import requests

from .config import PHONE_IP, PHONE_PORT

logger = logging.getLogger(__name__)


def trigger_emergency_sms(context: str) -> bool:
    """Send an emergency SMS via Automate HTTP server on the phone.

    Returns True if the request was delivered, False otherwise.
    """
    url = f"http://{PHONE_IP}:{PHONE_PORT}/"
    msg = f"BUDDY ALERT: {context}"
    try:
        requests.get(url, timeout=5)
        logger.info("Emergency SMS triggered: %s", msg)
        return True
    except requests.exceptions.ConnectionError:
        logger.warning("SMS trigger failed — phone unreachable at %s:%s", PHONE_IP, PHONE_PORT)
    except requests.exceptions.Timeout:
        logger.warning("SMS trigger timed out")
    except Exception as e:
        logger.warning("SMS trigger error: %s", e)
    return False
