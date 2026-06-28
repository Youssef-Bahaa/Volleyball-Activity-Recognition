import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T

_frame_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


WINDOW_SIZE = 9
NUM_PERSONS = 12
YOLO_CONF = 0.35
FRAME_STRIDE = 4

GROUP_LABELS = {
    0: "l-pass", 1: "r-pass", 2: "l-spike", 3: "r-spike",
    4: "l-set",  5: "r-set",  6: "l-winpoint", 7: "r-winpoint",
}
PERSON_LABELS = {
    0: "waiting", 1: "setting", 2: "digging", 3: "falling",
    4: "spiking", 5: "blocking", 6: "jumping", 7: "moving", 8: "standing",
}

# Black placeholder tensor for missing players
_BLANK_FRAME = torch.zeros(3, 224, 224)



class ActionPredictor:
    def __init__(self, models: dict, device: str):
        self.group_model = models["group_model"]
        self.yolo = models["yolo"]
        self.device = device

        self._buffers = [
            deque(maxlen=WINDOW_SIZE) for _ in range(NUM_PERSONS)
        ]

        self.frame_count  = 0
        self.since_infer  = 0
        self._last_group  = None
        self._last_conf   = 0.0

    def push_frame(self, frame_bgr):
        t0 = time.perf_counter()
        self.frame_count += 1
        self.since_infer += 1

        detections, player_tensors = self._detect_and_crop(frame_bgr)

        for pid in range(NUM_PERSONS):
            tensor = player_tensors[pid]
            self._buffers[pid].append(tensor)

        buf_len = len(self._buffers[0])
        ms = round((time.perf_counter() - t0) * 1000, 2)

        if buf_len < WINDOW_SIZE:
            return {
                "status": "collecting",
                "frame": self.frame_count,
                "buffer_size": buf_len,
                "total": WINDOW_SIZE,
                "group_label": None,
                "group_conf": None,
                "detections": detections,
                "persons": [],
                "ms": ms,
            }

        if self.since_infer >= FRAME_STRIDE:
            self.since_infer = 0
            group_label, group_conf, person_preds = self._infer(frame_bgr)
            self._last_group = group_label
            self._last_conf = group_conf
            self._last_persons = person_preds

        else:
            group_label  = self._last_group
            group_conf   = self._last_conf
            person_preds = getattr(self, "_last_persons", [])

        ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "status": "predicting",
            "frame": self.frame_count,
            "buffer_size": WINDOW_SIZE,
            "total": WINDOW_SIZE,
            "group_label": group_label,
            "group_conf": group_conf,
            "detections": detections,
            "persons": person_preds,
            "ms": ms,
        }

    def reset(self):
        for d in self._buffers:
            d.clear()
        self.frame_count = 0
        self.since_infer = 0
        self._last_group  = None
        self._last_conf   = 0.0
        self._last_persons = []

    def _detect_and_crop(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        results = self.yolo(frame_bgr, conf=YOLO_CONF, verbose=False)

        persons = []
        raw_dets = []

        for r in results:
            for box in r.boxes:
                label = r.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
                raw_dets.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "box": [x1, y1, x2, y2],
                })
                if label == "person":
                    persons.append((x1, y1, x2, y2, conf))

        persons.sort(key=lambda p: (p[0] + p[2]) / 2)

        player_tensors = []
        for pid in range(NUM_PERSONS):
            if pid < len(persons):
                x1, y1, x2, y2, _ = persons[pid]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    player_tensors.append(_BLANK_FRAME)
                else:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    player_tensors.append(_frame_transform(crop_rgb))
            else:
                player_tensors.append(_BLANK_FRAME)

        return raw_dets, player_tensors

    def _infer(self, frame_bgr: np.ndarray):
        player_clips = []
        for pid in range(NUM_PERSONS):
            frames = list(self._buffers[pid])          # T tensors (3,224,224)
            clip   = torch.stack(frames)               # (T, 3, 224, 224)
            player_clips.append(clip)

        batch = torch.stack(player_clips).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.group_model(batch)
            probs  = torch.softmax(logits, dim=-1)[0]
            idx  = int(probs.argmax())
            conf = round(float(probs[idx]), 4)
            group_label = GROUP_LABELS.get(idx, str(idx))
            person_preds = self._infer_person_actions(batch)

        return group_label, conf, person_preds

    def _infer_person_actions(self, x):
        b, n, t, c, h, w = x.shape
        results = []

        with torch.no_grad():
            resnet = self.group_model.resnet50
            lstm1  = self.group_model.lstm1

            x = x.view(b * n * t, c, h, w)

            x = x.view(b * n * t, c, h, w)
            feats = resnet(x)  # (b*n*t, 2048, 1, 1)
            feats = feats.view(b * n, t, -1)  # (b*n, t, 2048)
            out, _ = lstm1(feats)  # (b*n, t, 512)
            last = out[:, -1, :]  # (b*n, 512) — last timestep

            for pid in range(n):
                results.append({
                    "player_id": pid,
                    "action_label": None,
                    "confidence": None,
                })

        return results






