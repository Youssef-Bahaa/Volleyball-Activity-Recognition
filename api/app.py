import json
import uuid
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.model_loader import load_all
from api.predictor import ActionPredictor

UPLOAD_DIR = Path("tmp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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
            "B8_Group": "loaded" if _models.get("group_model") else "not loaded",
            "YOLO": "loaded" if _models.get("yolo") else "not loaded",
        },
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {'.mp4', '.avi', '.mov', '.mkv'}:
        return JSONResponse({"error": f"Unsupported format: {suffix}"}, status_code=400)

    clip_id  = uuid.uuid4().hex
    clip_dir = UPLOAD_DIR / clip_id
    clip_dir.mkdir()

    try:
        video_path = clip_dir / f"video{suffix}"
        with open(video_path, 'wb') as f:
            f.write(await file.read())

        timeline = _process_video(str(video_path))
        return JSONResponse({"timeline": timeline})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        shutil.rmtree(clip_dir, ignore_errors=True)


def _process_video(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    SAMPLE_EVERY = 3
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % SAMPLE_EVERY == 0:
            frames.append(frame)
        idx += 1
    cap.release()

    effective_fps = fps / SAMPLE_EVERY
    WINDOW = 9
    STRIDE = int(effective_fps)
    total = len(frames)

    if total < WINDOW:
        return []

    timeline = []
    predictor = ActionPredictor(_models, _device)
    last_label = None

    start = 0
    while start + WINDOW <= total:
        predictor.reset()
        result = None

        for frame in frames[start: start + WINDOW]:
            result = predictor.push_frame(frame)

        if result and result["group_label"] is not None:
            label = result["group_label"]
            if label != last_label:
                timeline.append({
                    "start": _ts(start, effective_fps),
                    "end":  _ts(start + WINDOW, effective_fps),
                    "start_sec": round(start / effective_fps, 2),
                    "end_sec": round((start + WINDOW) / effective_fps, 2),
                    "label": label,
                    "confidence": round(result["group_conf"], 3),
                })
                last_label = label

        start += STRIDE

    return timeline


def _ts(frame_idx: int, fps: float) -> str:
    s = int(frame_idx / fps)
    return f"{s // 60}:{s % 60:02d}"


if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)