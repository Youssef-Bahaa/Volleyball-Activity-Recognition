import time
from collections import deque

import cv2
import numpy as np
import torch
import torchvision.transforms as T

_frame_transform=T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    ),
])

WINDOW_SIZE=9
NUM_PERSONS=12
YOLO_CONF=0.20
YOLO_IOU=0.45
FRAME_STRIDE=4

GROUP_LABELS={
    0:"l-pass",
    1:"r-pass",
    2:"l-spike",
    3:"r-spike",
    4:"l-set",
    5:"r-set",
    6:"l-winpoint",
    7:"r-winpoint",
}

_BLANK_FRAME=torch.zeros(3,224,224)

class ActionPredictor:
    def __init__(self,models:dict,device:str):
        self.group_model=models["group_model"]
        self.yolo=models["yolo"]
        self.device=device

        self._buffers=[
            deque(maxlen=WINDOW_SIZE)
            for _ in range(NUM_PERSONS)
        ]

        self.frame_count=0
        self.since_infer=0
        self._last_group=None
        self._last_conf=0.0
        self._last_player_count=0

    def push_frame(self,frame_bgr:np.ndarray)->dict:
        t0=time.perf_counter()

        self.frame_count+=1
        self.since_infer+=1

        player_tensors,player_count=self._detect_and_crop(frame_bgr)

        for pid in range(NUM_PERSONS):
            self._buffers[pid].append(player_tensors[pid])

        buf_len=len(self._buffers[0])

        ms=round(
            (time.perf_counter()-t0)*1000,
            2
        )

        if buf_len<WINDOW_SIZE:
            return{
                "status":"collecting",
                "frame":self.frame_count,
                "buffer_size":buf_len,
                "total":WINDOW_SIZE,
                "group_label":None,
                "group_conf":None,
                "player_count":player_count,
                "ms":ms,
            }

        if self.since_infer>=FRAME_STRIDE:
            self.since_infer=0

            group_label,group_conf=self._infer()

            self._last_group=group_label
            self._last_conf=group_conf
            self._last_player_count=player_count

        else:
            group_label=self._last_group
            group_conf=self._last_conf
            player_count=self._last_player_count

        ms=round(
            (time.perf_counter()-t0)*1000,
            2
        )

        return{
            "status":"predicting",
            "frame":self.frame_count,
            "buffer_size":WINDOW_SIZE,
            "total":WINDOW_SIZE,
            "group_label":group_label,
            "group_conf":group_conf,
            "player_count":player_count,
            "ms":ms,
        }

    def reset(self):
        for d in self._buffers:
            d.clear()

        self.frame_count=0
        self.since_infer=0
        self._last_group=None
        self._last_conf=0.0
        self._last_player_count=0

    def _detect_and_crop(self,frame_bgr:np.ndarray):
        h,w=frame_bgr.shape[:2]

        results=self.yolo(
            frame_bgr,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            verbose=False
        )

        persons=[]

        for r in results:
            for box in r.boxes:
                if r.names[int(box.cls)]=="person":
                    x1,y1,x2,y2=[
                        round(v)
                        for v in box.xyxy[0].tolist()
                    ]

                    conf=float(box.conf)

                    persons.append(
                        (x1,y1,x2,y2,conf)
                    )

        persons.sort(
            key=lambda p:(p[0]+p[2])/2
        )

        player_tensors=[]

        for pid in range(NUM_PERSONS):
            if pid<len(persons):
                x1,y1,x2,y2,_=persons[pid]

                x1=max(0,x1)
                y1=max(0,y1)
                x2=min(w,x2)
                y2=min(h,y2)

                crop=frame_bgr[y1:y2,x1:x2]

                if crop.size==0:
                    player_tensors.append(
                        _BLANK_FRAME
                    )
                else:
                    crop_rgb=cv2.cvtColor(
                        crop,
                        cv2.COLOR_BGR2RGB
                    )

                    player_tensors.append(
                        _frame_transform(crop_rgb)
                    )
            else:
                player_tensors.append(
                    _BLANK_FRAME
                )

        return player_tensors,len(persons)

    def _infer(self):
        player_clips=[]

        for pid in range(NUM_PERSONS):
            frames=list(self._buffers[pid])
            clip=torch.stack(frames)
            player_clips.append(clip)

        batch=torch.stack(
            player_clips
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits=self.group_model(batch)

            probs=torch.softmax(
                logits,
                dim=-1
            )[0]

            idx=int(probs.argmax())
            conf=round(float(probs[idx]),4)

            group_label=GROUP_LABELS.get(
                idx,
                str(idx)
            )

        return group_label,conf