import os
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.dataset.transforms import train_transform, val_transform, test_transform
from src.dataset.boxinfo import BoxInfo
from src.dataset.utils import PERSON_ACTION_TO_ID, filter_by_ids  # ← import, don't redefine
import sys
import src.dataset.boxinfo as _boxinfo_mod

sys.modules.setdefault('boxinfo', _boxinfo_mod)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'BoxInfo':
            return BoxInfo
        return super().find_class(module, name)


class PersonTempDataset(Dataset):
    def __init__(self, image_root, annot_path, transform=None):
        self.samples = []
        self.transform = transform

        with open(annot_path, 'rb') as f:
            data = _Unpickler(f).load()

        for video_id, clips in data.items():
            for clip_id, clip_info in clips.items():
                frame_boxes = clip_info['frame_boxes_dct']
                sorted_frames = sorted(frame_boxes.keys())

                img_paths = [
                    os.path.join(image_root, video_id, clip_id, f"{fid}.jpg")
                    for fid in sorted_frames
                    if os.path.exists(os.path.join(image_root, video_id, clip_id, f"{fid}.jpg"))
                ]

                if not img_paths:
                    continue

                self.samples.append((video_id, clip_id, img_paths, sorted_frames, frame_boxes))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, clip_id, img_paths, sorted_frames, frame_boxes = self.samples[idx]

        num_persons = 12
        persons_data = [[] for _ in range(num_persons)]
        labels = [None] * num_persons

        for t, img_path in enumerate(img_paths):
            fid = sorted_frames[t]
            boxes = frame_boxes[fid]

            img = Image.open(img_path).convert('RGB')

            box_dict = {b.player_ID: b for b in boxes}

            for pid in range(num_persons):
                box_info = box_dict.get(pid)

                if box_info is None:
                    dummy = self.transform(img) if self.transform else img
                    persons_data[pid].append(torch.zeros_like(dummy))
                else:
                    if labels[pid] is None:
                        labels[pid] = PERSON_ACTION_TO_ID.get(box_info.action.lower(), 0)  # ← use imported dict directly

                    x1, y1, x2, y2 = box_info.box
                    crop = img.crop((x1, y1, x2, y2))
                    crop = self.transform(crop) if self.transform else crop
                    persons_data[pid].append(crop)

        labels = [l if l is not None else 0 for l in labels]

        video_tensor = torch.stack([torch.stack(p) for p in persons_data])  # (N, T, C, H, W)
        labels = torch.tensor(labels, dtype=torch.long)

        return video_id, clip_id, video_tensor, labels


def collate_fn(batch):
    video_ids, clip_ids, tensors, labels = zip(*batch)
    tensors = torch.stack(tensors)   # (B, N, T, C, H, W)
    labels  = torch.stack(labels)    # (B, N)
    B, N = labels.shape
    labels = labels.view(B * N)     # (B*N,)
    return video_ids, clip_ids, tensors, labels

def build_loaders(cfg):
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]

    train_dataset = PersonTempDataset(
        data_cfg["videos_path"],
        data_cfg["annot_path"],
        transform=train_transform
    )

    val_dataset = PersonTempDataset(
        data_cfg["videos_path"],
        data_cfg["annot_path"],
        transform=val_transform
    )

    test_dataset = PersonTempDataset(
        data_cfg["videos_path"],
        data_cfg["annot_path"],
        transform=test_transform
    )

    train_dataset = filter_by_ids(train_dataset, data_cfg["video_splits"]["train"])
    val_dataset   = filter_by_ids(val_dataset, data_cfg["video_splits"]["validation"])
    test_dataset  = filter_by_ids(test_dataset, data_cfg["video_splits"]["test"])

    loader_kwargs = {
        "batch_size": training_cfg["batch_size"],
        "num_workers": training_cfg.get("num_workers", 4),
        "pin_memory": training_cfg.get("pin_memory", True),
        "collate_fn": collate_fn,
    }

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader