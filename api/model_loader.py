import torch
from ultralytics import YOLO
from src.models.B8.Person_Temporal import PersonTemp
from src.models.B8.Group_Temporal import GroupActivityB8
from src.utils.checkpoint import CheckpointManager
from src.utils.paths import Paths


YOLO_MODEL = "yolov8n.pt"
YOLO_CONF  = 0.40

def load_person_model(device: str) -> PersonTemp:
    model = PersonTemp()

    try:
        p = Paths('.', model_name = 'B8_Person')
        ckpt_path = p.best_checkpoint()
        CheckpointManager.load(ckpt_path, model, device=device)
        print(r'Person model loaded from path {ckpt_path}')
    except FileNotFoundError:
        print("No Checkpoint Found!")

    model.to(device).eval()
    return model

def load_group_model(person_model: PersonTemp, device: str) -> GroupActivityB8:
    model = GroupActivityB8(player_model=person_model)
    try:
        p = Paths('.', model_name = 'B8_Group')
        ckpt_path = p.best_checkpoint()
        CheckpointManager.load(ckpt_path, model, device=device)
        print(r'Group model loaded from path {ckpt_path}')
    except FileNotFoundError:
        print("No Checkpoint Found!")

    model.to(device).eval()
    return model



def load_yolo(device: str) -> YOLO:
    yolo = YOLO(YOLO_MODEL)
    yolo.to(device)
    return yolo


def load_all(device: str) -> dict:
    print('Loading All Models')
    person_model = load_person_model(device=device)
    group_model = load_group_model(person_model=person_model, device=device)
    yolo = load_yolo(device=device)
    print('All models ready')
    return {
        'person_model': person_model,
        'group_model' : group_model,
        'yolo' : yolo,
        'device' : device
    }







