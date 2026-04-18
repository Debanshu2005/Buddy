"""
Test sms_service module.

Run from pi_services/:
    python test_sms_service.py

Set env vars to point at your phone before running:
    export BUDDY_PHONE_IP=192.168.1.5
    export BUDDY_PHONE_PORT=8080
"""

from unittest.mock import MagicMock, patch

from sms_service import trigger_emergency_sms


def test_success():
    with patch("sms_service.core.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        result = trigger_emergency_sms("Fall detected in living room")
        assert result is True
        mock_get.assert_called_once()
        url, kwargs = mock_get.call_args[0][0], mock_get.call_args[1]
        assert "/emergency" in url
        assert "BUDDY ALERT" in kwargs["params"]["msg"]
    print("PASS test_success")


def test_connection_error():
    with patch("sms_service.core.requests.get", side_effect=Exception("unreachable")):
        result = trigger_emergency_sms("Test context")
        assert result is False
    print("PASS test_connection_error")


def test_message_format():
    with patch("sms_service.core.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        trigger_emergency_sms("smoke detected")
        sent_msg = mock_get.call_args[1]["params"]["msg"]
        assert sent_msg == "BUDDY ALERT: smoke detected"
    print("PASS test_message_format")


if __name__ == "__main__":
    test_success()
    test_connection_error()
    test_message_format()
    print("\nAll tests passed.")
