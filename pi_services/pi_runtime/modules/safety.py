"""Safety mixin — fall/blood/behavior monitors, wellness checks."""
from __future__ import annotations
import os, threading, time
from typing import Optional
import cv2
import numpy as np
from hardware.oled_eyes import EyeState

class SafetyMixin:
    def _on_behavior_alert(self, result: dict):
        """Called by behavior pipeline on risk detection — speaks alert and sends WhatsApp."""
        decision = result.get("decision", {})
        label = decision.get("label", "unknown")
        severity = decision.get("severity", "low")
        reason = decision.get("reason", "")
        print(f"[Behavior] 🚨 ALERT: {label} | {severity} | {reason}")

        jpeg_bytes = None
        with self._stream_lock:
            jpeg_bytes = self._stream_frame

        if not self.is_speaking:
            msg = {
                "fall_detected": "Warning! Someone may have fallen!",
                "possible_medical_emergency": "Emergency! Someone may need medical help!",
                "injury_suspected": "Alert! Someone appears to be injured.",
                "monitor_closely": "Someone has been inactive for a while.",
            }.get(label, f"Behavior alert: {label}")
            self.speak(msg)

        self._send_alert_channels(
            label=label,
            severity=severity,
            reason=reason,
            jpeg_bytes=jpeg_bytes,
        )


    def _on_surveillance_event(
        self,
        event_type: str,
        description: str,
        confidence: float,
        severity: str,
    ) -> None:
        """
        Callback fired by _SurveillanceClient for every new MediaPipe event
        detected on the PC surveillance server.

        Routing:
          critical  → full emergency response (Telegram + webhook + call command)
          high      → alert channels only (Telegram)
          medium    → spoken warning only
        """
        if self.sleep_mode or self._emergency_active:
            return

        reason = (
            f"PC surveillance detected: {event_type} — {description} "
            f"(confidence {confidence:.0%})"
        )
        print(f"[Surveillance] {severity.upper()} received on Pi: {event_type} | {description} ({confidence:.0%})")

        jpeg_bytes: Optional[bytes] = None
        with self._stream_lock:
            jpeg_bytes = self._stream_frame

        # Spoken message map
        _spoken: dict[str, str] = {
            "fall":                  "Warning! Someone may have fallen!",
            "person_down":           "Alert! A person appears to be lying on the floor.",
            "hand_on_chest":         "Alert! Someone has their hand on their chest. Are you okay?",
            "eyes_closed":           "Alert! Someone's eyes have been closed for a while.",
            "trembling":             "Alert! Trembling detected. Are you okay?",
            "head_tilt":             "Alert! Unusual head tilt detected. Please check.",
            "hands_raised":          "I noticed your hands are raised. Do you need help?",
            "prolonged_inactivity":  "I haven't seen you move for a while. Are you okay?",
            "covering_face":         "I noticed you are covering your face. Are you alright?",
            "head_drooping":         "Your head appears to be drooping. Are you okay?",
            "clutching_stomach":     "I noticed you may be holding your stomach. Are you okay?",
            "hunching":              "You seem to be hunching. Are you in pain?",
            "crouching":             "I see you crouching. Is everything alright?",
        }
        spoken_msg = _spoken.get(event_type, f"Surveillance alert: {description}")

        if severity == "critical":
            if not self.is_speaking:
                self.speak(spoken_msg)
            self._trigger_emergency_response(reason)

        elif severity == "high":
            if not self.is_speaking:
                self.speak(spoken_msg)
            self._send_alert_channels(
                label=event_type,
                severity=severity,
                reason=reason,
                jpeg_bytes=jpeg_bytes,
            )

        else:  # medium
            if not self.is_speaking:
                self.speak(spoken_msg)


    def _person_or_face_visible(self) -> bool:
        if self._get_faces_snapshot():
            return True
        return any(
            isinstance(det, dict) and det.get("name") == "person"
            for det in self._get_detections_snapshot()
        )


    def _largest_person_bbox(self) -> Optional[tuple[float, float, float, float]]:
        person_boxes = []
        for det in self._get_detections_snapshot():
            if not isinstance(det, dict) or det.get("name") != "person":
                continue
            box = det.get("bbox") or det.get("box") or det.get("xyxy")
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            area = width * height
            if area > 0:
                person_boxes.append((area, x1, y1, x2, y2))
        if not person_boxes:
            return None
        _, x1, y1, x2, y2 = max(person_boxes, key=lambda item: item[0])
        return x1, y1, x2, y2


    def _update_bbox_fall_monitor(self, frame: np.ndarray):
        if os.getenv("BUDDY_ENABLE_BBOX_FALL_DETECTION", "1") == "0":
            return
        if self._emergency_active:
            return

        bbox = self._largest_person_bbox()
        if bbox is None:
            self._bbox_fall_started_at = None
            return

        frame_h, frame_w = frame.shape[:2]
        frame_area = float(max(1, frame_w * frame_h))
        x1, y1, x2, y2 = bbox
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        aspect_ratio = width / height
        area_ratio = (width * height) / frame_area
        center_y_ratio = ((y1 + y2) / 2.0) / max(1.0, float(frame_h))

        fall_aspect = float(os.getenv("BUDDY_BBOX_FALL_ASPECT_RATIO", self._BBOX_FALL_ASPECT_RATIO))
        min_area = float(os.getenv("BUDDY_BBOX_FALL_MIN_AREA_RATIO", self._BBOX_FALL_MIN_AREA_RATIO))
        low_center = float(os.getenv("BUDDY_BBOX_FALL_LOW_CENTER_RATIO", self._BBOX_FALL_LOW_CENTER_RATIO))
        confirm_seconds = float(os.getenv("BUDDY_BBOX_FALL_CONFIRM_SECONDS", self._BBOX_FALL_CONFIRM_SECONDS))
        cooldown = float(os.getenv("BUDDY_BBOX_FALL_ALERT_COOLDOWN_SECONDS", self._BBOX_FALL_ALERT_COOLDOWN_SECONDS))

        looks_fallen = (
            aspect_ratio >= fall_aspect
            and area_ratio >= min_area
            and center_y_ratio >= low_center
        )
        now = time.time()
        if not looks_fallen:
            self._bbox_fall_started_at = None
            return

        if self._bbox_fall_started_at is None:
            self._bbox_fall_started_at = now
            return

        if now - self._bbox_fall_started_at < confirm_seconds:
            return
        if now - self._bbox_fall_last_alert_time < cooldown:
            return

        self._bbox_fall_last_alert_time = now
        reason = (
            "Pi-safe bbox fall detector: person appears horizontal/low "
            f"for {now - self._bbox_fall_started_at:.1f}s "
            f"(aspect={aspect_ratio:.2f}, area={area_ratio:.2f}, center_y={center_y_ratio:.2f})"
        )
        print(f"[Behavior] BBox fall alert: {reason}")
        jpeg_bytes = None
        with self._stream_lock:
            jpeg_bytes = self._stream_frame

        if not self.is_speaking:
            self.speak("Warning. Someone may have fallen.")

        self._send_alert_channels(
            label="fall_detected",
            severity="high",
            reason=reason,
            jpeg_bytes=jpeg_bytes,
        )


    def _update_blood_monitor(self, frame: np.ndarray):
        """Detect blood-like red pooling as a low-confidence visual risk signal."""
        if os.getenv("BUDDY_ENABLE_BLOOD_HEURISTIC", "1") == "0":
            return
        if self._emergency_active:
            return

        roi = frame
        bbox = self._largest_person_bbox()
        if bbox is not None:
            frame_h, frame_w = frame.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            pad_x = int((x2 - x1) * 0.25)
            pad_y = int((y2 - y1) * 0.25)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(frame_w, x2 + pad_x)
            y2 = min(frame_h, y2 + pad_y)
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            self._blood_detect_started_at = None
            return

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat_min = int(os.getenv("BUDDY_BLOOD_MIN_SATURATION", str(self._BLOOD_MIN_SATURATION)))
        val_min = int(os.getenv("BUDDY_BLOOD_MIN_VALUE", str(self._BLOOD_MIN_VALUE)))

        lower_red_1 = np.array([0, sat_min, val_min], dtype=np.uint8)
        upper_red_1 = np.array([12, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([165, sat_min, val_min], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_red_1, upper_red_1) | cv2.inRange(hsv, lower_red_2, upper_red_2)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        red_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour_area = max((cv2.contourArea(contour) for contour in contours), default=0.0)
        contour_ratio = largest_contour_area / float(max(1, roi.shape[0] * roi.shape[1]))
        redness_strength = float(np.mean(hsv[:, :, 1][mask > 0])) if np.any(mask) else 0.0

        min_ratio = float(os.getenv("BUDDY_BLOOD_MIN_REGION_RATIO", str(self._BLOOD_MIN_REGION_RATIO)))
        looks_blood_like = (
            red_ratio >= min_ratio
            and contour_ratio >= (min_ratio * 0.4)
            and redness_strength >= max(sat_min, 100)
            and self._person_or_face_visible()
        )

        now = time.time()
        if not looks_blood_like:
            self._blood_detect_started_at = None
            return

        if self._blood_detect_started_at is None:
            self._blood_detect_started_at = now
            return

        confirm_seconds = float(os.getenv("BUDDY_BLOOD_CONFIRM_SECONDS", str(self._BLOOD_CONFIRM_SECONDS)))
        cooldown = float(os.getenv("BUDDY_BLOOD_ALERT_COOLDOWN_SECONDS", str(self._BLOOD_ALERT_COOLDOWN_SECONDS)))
        if (now - self._blood_detect_started_at) < confirm_seconds:
            return
        if (now - self._blood_last_alert_time) < cooldown:
            return

        self._blood_last_alert_time = now
        reason = (
            "Experimental blood-like scene heuristic triggered "
            f"(red_ratio={red_ratio:.3f}, contour_ratio={contour_ratio:.3f}, "
            f"redness={redness_strength:.1f}). This is not a medical diagnosis."
        )
        print(f"[Behavior] Blood-like alert: {reason}")
        jpeg_bytes = None
        with self._stream_lock:
            jpeg_bytes = self._stream_frame

        if not self.is_speaking:
            self.speak("Warning. I can see a possible blood-like scene. Please check immediately.")

        self._send_alert_channels(
            label="possible_blood_detected",
            severity="high",
            reason=reason,
            jpeg_bytes=jpeg_bytes,
        )


    def _update_behavior_monitor(self, frame: np.ndarray):
        if os.getenv("BUDDY_ENABLE_VISUAL_SAFETY", "1") == "0":
            return
        if self._emergency_active:
            return

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
        except Exception as exc:
            self.logger.warning("Behavior monitor frame prep failed: %s", exc)
            return

        if self._previous_behavior_frame is None:
            self._previous_behavior_frame = gray
            self._last_motion_time = time.time()
            return

        diff = cv2.absdiff(self._previous_behavior_frame, gray)
        motion_score = float(np.mean(diff))
        self._previous_behavior_frame = gray

        if motion_score >= float(os.getenv("BUDDY_VISUAL_MOTION_THRESHOLD", self._VISUAL_MOTION_THRESHOLD)):
            self._last_motion_time = time.time()

        if not self._person_or_face_visible():
            self._last_motion_time = time.time()
            return

        now = time.time()
        still_for = now - self._last_motion_time
        stillness_limit = float(os.getenv("BUDDY_VISUAL_STILLNESS_SECONDS", self._VISUAL_STILLNESS_SECONDS))
        cooldown = float(os.getenv("BUDDY_VISUAL_CHECK_COOLDOWN_SECONDS", self._VISUAL_CHECK_COOLDOWN_SECONDS))

        if still_for < stillness_limit:
            return
        if self._visual_check_active or (now - self._last_visual_check_time) < cooldown:
            return

        reason = f"Person or face visible with very little movement for {still_for:.0f} seconds"
        self._visual_check_active = True
        self._last_visual_check_time = now
        threading.Thread(
            target=self._visual_wellness_check,
            args=(reason,),
            daemon=True,
        ).start()


    def _visual_wellness_check(self, reason: str):
        self._pause_wake_listening = True
        try:
            self.logger.warning("Visual safety check: %s", reason)
            self.speak("I have not seen you move for a while. Are you okay?")
            self._wait_for_tts()
            self._play_listen_beep()
            self._eye(EyeState.LISTENING)
            answer = self.listen_for_speech_with_initial_timeout(10.0)
            self._eye(EyeState.IDLE)

            if answer and self._is_ok_response(answer):
                self.speak("Okay. I am glad you are alright.")
                self._last_motion_time = time.time()
                return

            if not answer or self._is_not_ok_response(answer):
                self._trigger_emergency_response(answer or reason)
                return

            self.speak("I am not sure I understood. I will stay alert. If you need help, say emergency.")
        finally:
            self._visual_check_active = False
            self._pause_wake_listening = False


