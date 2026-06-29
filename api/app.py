import uuid
import shutil
from pathlib import Path

import cv2
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.model_loader import load_all
from api.predictor import ActionPredictor

UPLOAD_DIR=Path("tmp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app=FastAPI(title="Volleyball Activity Recognizer",version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_device="cuda" if torch.cuda.is_available() else "cpu"
_models={}

@app.on_event("startup")
def on_startup():
    global _models
    _models=load_all(_device)

@app.get("/health")
def health():
    return{
        "status":"ok",
        "device":_device,
        "models":{
            "B8_Group":"loaded" if _models.get("group_model") else "not loaded",
            "YOLO":"loaded" if _models.get("yolo") else "not loaded",
        },
    }

@app.post("/predict")
async def predict(file:UploadFile=File(...)):
    suffix=Path(file.filename).suffix.lower()

    if suffix not in {".mp4",".avi",".mov",".mkv"}:
        return JSONResponse(
            {"error":f"Unsupported format: {suffix}"},
            status_code=400
        )

    clip_id=uuid.uuid4().hex
    clip_dir=UPLOAD_DIR/clip_id
    clip_dir.mkdir()

    try:
        video_path=clip_dir/f"video{suffix}"

        with open(video_path,"wb") as f:
            f.write(await file.read())

        result=_process_video(str(video_path))
        return JSONResponse(result)

    except Exception as e:
        return JSONResponse(
            {"error":str(e)},
            status_code=500
        )

    finally:
        shutil.rmtree(clip_dir,ignore_errors=True)

def _process_video(video_path:str)->dict:
    cap=cv2.VideoCapture(video_path)
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0

    SAMPLE_EVERY=3
    frames=[]
    timestamps=[]
    idx=0

    while True:
        ret,frame=cap.read()

        if not ret:
            break

        if idx%SAMPLE_EVERY==0:
            frames.append(frame)
            timestamps.append(idx/fps)

        idx+=1

    cap.release()

    effective_fps=fps/SAMPLE_EVERY
    WINDOW=9
    total=len(frames)

    if total<WINDOW:
        return{
            "frames":[],
            "effective_fps":effective_fps
        }

    predictor=ActionPredictor(_models,_device)
    frames_out=[]

    for i,(frame,ts) in enumerate(zip(frames,timestamps)):
        result=predictor.push_frame(frame)

        frames_out.append({
            "frame_idx":i,
            "timestamp":round(ts,3),
            "group_label":result.get("group_label"),
            "group_conf":result.get("group_conf"),
            "player_count":result.get("player_count",0),
        })

    return{
        "frames":frames_out,
        "effective_fps":round(effective_fps,4),
    }

if __name__=="__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )