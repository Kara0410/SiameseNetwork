"""WebSocket endpoint for real-time webcam verification.

The client streams base64-encoded JPEG frames as `{"image": "data:image/jpeg;base64,...",
"reset": false}`. The first frame (or any frame sent with `"reset": true`) becomes the
reference embedding; subsequent frames are compared against it and run through the
liveness heuristics, giving a live similarity + anti-spoofing readout.
"""

import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ai_services.embeddings import get_model
from ai_services.liveness import assess_liveness
from ai_services.preprocessing import load_image_from_base64, resize_max_dim

from ...core.config import get_settings

router = APIRouter()


@router.websocket("/live")
async def live(websocket: WebSocket) -> None:
    await websocket.accept()

    settings = get_settings()
    model = get_model(settings.embedding_default)
    reference_vector: np.ndarray | None = None

    try:
        while True:
            payload = await websocket.receive_json()
            image_data = payload.get("image")
            if not image_data:
                await websocket.send_json({"error": "missing 'image' field"})
                continue

            start = time.perf_counter()
            image = resize_max_dim(load_image_from_base64(image_data))
            vector = model.embed(image)
            liveness = assess_liveness(image)

            if reference_vector is None or payload.get("reset"):
                reference_vector = vector
                similarity = 1.0
            else:
                similarity = float(np.dot(vector, reference_vector))

            latency_ms = (time.perf_counter() - start) * 1000

            await websocket.send_json(
                {
                    "model": settings.embedding_default,
                    "similarity": similarity,
                    "spoof_risk": liveness.risk,
                    "flags": liveness.flags,
                    "latency_ms": latency_ms,
                }
            )
    except WebSocketDisconnect:
        return
