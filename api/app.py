import json

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.model_loader import load_all
from api.predictor import ActionPredictor


app = FastAPI(title="Volleyball Activity Recognizer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_device = "cuda" if torch.cuda.is_available() else "cpu"
_models: dict = {}


@app.on_event("startup")
def on_startup():
    global _models
    _models = load_all(_device)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": _device,
        "models": {
            "B8_Person":"loaded" if _models.get("person_model") else "not loaded",
            "B8_Group": "loaded" if _models.get("group_model")  else "not loaded",
            "YOLO": "loaded" if _models.get("yolo")  else "not loaded",
        },
    }


@app.websocket("/ws/predict")
async def predict_stream(websocket: WebSocket):
    """
    One connection = one video session.
    Each session gets its own ActionPredictor with its own frame buffers.

    Frame flow:
      1. Browser grabs frame from <video> via <canvas>
      2. Converts to JPEG blob → sends raw bytes over WebSocket
      3. Server decodes JPEG → BGR numpy
      4. Pushes to ActionPredictor.push_frame()
      5. Sends back JSON result
    """
    await websocket.accept()
    print(f"[ws] Client connected  device={_device}")

    predictor = ActionPredictor(_models, _device)

    try:
        while True:
            raw = await websocket.receive_bytes()

            frame = _decode_jpeg(raw)
            if frame is None:
                await websocket.send_text(json.dumps({"error": "failed to decode frame"}))
                continue

            result = predictor.push_frame(frame)
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as exc:
        print(f"Error: {exc}")
        try:
            await websocket.send_text(json.dumps({"error": str(exc)}))
        except Exception:
            pass
    finally:
        predictor.reset()


def _decode_jpeg(data: bytes):
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)




