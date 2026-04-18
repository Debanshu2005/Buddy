"""FastAPI integration for lightweight behavior-risk detection."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import logging
from threading import Lock
import time
from typing import Optional

import cv2
import numpy as np

try:
    from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
except Exception:  # pragma: no cover - optional dependency in some dev setups
    APIRouter = None
    FastAPI = None
    File = Form = UploadFile = None
    HTTPException = RuntimeError

from .pipeline import BehaviorDetectionPipeline, PipelineConfig


LOGGER = logging.getLogger(__name__)
router = APIRouter() if APIRouter is not None else None


class PipelineRegistry:
    """Keep per-session pipelines so temporal features survive between requests."""

    def __init__(self) -> None:
        self._pipelines: dict[str, BehaviorDetectionPipeline] = {}
        self._last_seen: dict[str, float] = defaultdict(float)
        self._lock = Lock()

    def get(self, session_id: str, config: Optional[PipelineConfig] = None) -> BehaviorDetectionPipeline:
        """Create or reuse a session-local pipeline instance."""
        with self._lock:
            if session_id not in self._pipelines:
                self._pipelines[session_id] = BehaviorDetectionPipeline(config=config)
            self._last_seen[session_id] = time.time()
            self._evict_idle(max_idle_seconds=600)
            return self._pipelines[session_id]

    def _evict_idle(self, max_idle_seconds: float) -> None:
        now = time.time()
        expired = [session_id for session_id, last_seen in self._last_seen.items() if now - last_seen > max_idle_seconds]
        for session_id in expired:
            self._pipelines.pop(session_id, None)
            self._last_seen.pop(session_id, None)


REGISTRY = PipelineRegistry()


if router is not None:

    @router.get("/detect_behavior/health")
    async def behavior_health() -> dict:
        """Health endpoint for deployment probes."""
        return {"status": "ok", "service": "behavior_detection"}


    @router.post("/detect_behavior")
    async def detect_behavior(
        frame: UploadFile = File(...),
        session_id: str = Form("default"),
        process_every_n_frames: int = Form(2),
        enable_visualization: bool = Form(False),
    ) -> dict:
        """Run pose-based risk detection on an uploaded frame."""
        if not frame.content_type or not frame.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Expected an image upload.")

        payload = await frame.read()
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise HTTPException(status_code=400, detail="Could not decode image payload.")

        config = PipelineConfig(
            process_every_n_frames=max(process_every_n_frames, 1),
            enable_visualization=enable_visualization,
        )
        pipeline = REGISTRY.get(session_id=session_id, config=config)
        return pipeline.process_frame(decoded)


def create_app() -> "FastAPI":
    """Create a standalone FastAPI app exposing the behavior endpoint."""
    if FastAPI is None or router is None:
        raise RuntimeError("FastAPI is not installed. Add fastapi and uvicorn to run the API service.")
    app = FastAPI(title="Buddy Behavior Detection API")
    app.include_router(router)
    return app


app = create_app() if FastAPI is not None and router is not None else None
