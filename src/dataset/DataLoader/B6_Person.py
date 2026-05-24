import os
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.dataset.transforms import train_transform, val_transform, test_transform
from src.dataset.utils import activity2id, TransformSubset, filter_by_ids, PERSON_ACTION_TO_ID
import sys
import src.dataset.boxinfo as boxinfo

sys.modules["boxinfo"] = boxinfo


class VolleyBallPersonDataset(Dataset):
    def __init__(self, image_root, annot_path):
        self.samples = []

        with open(annot_path, 'rb') as f:
            data = pickle.load(f)

        for video_id in os.listdir(image_root):
            video_dir = os.path.join(image_root, video_id)
            if not os.path.isdir(video_dir):
                continue

            for clip_id in os.listdir(video_dir):
                clip_dir = os.path.join(video_dir, clip_id)
                if not os.path.isdir(clip_dir):
                    continue

                try:
                    clip_info = data[video_id][clip_id]
                    frame_boxes = clip_info['frame_boxes_dct']
                except KeyError:
                    continue

                for fid, boxes in frame_boxes.items():
                    img_path = os.path.join(clip_dir, f'{fid}.jpg')
                    if not os.path.exists(img_path):
                        continue

                    for box_info in boxes:
                        action = box_info.action.strip()
                        if action not in PERSON_ACTION_TO_ID:
                            continue

                        self.samples.append((
                            video_id,
                            img_path,
                            box_info.box,                  # (x1, y1, x2, y2)
                            PERSON_ACTION_TO_ID[action]
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, img_path, box, label = self.samples[idx]
        x1, y1, x2, y2 = box
        img = Image.open(img_path).convert('RGB')
        crop = img.crop((x1, y1, x2, y2))
        return video_id, crop, torch.tensor(label, dtype=torch.long)



class PersonTransformSubset(Dataset):
    """Applies transform to the person crop returned by VolleyBallPersonDataset."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        video_id, crop, label = self.subset[idx]
        return self.transform(crop), label



def build_person_loaders(cfg):
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]

    full_dataset = VolleyBallPersonDataset(
        image_root=data_cfg["videos_path"],
        annot_path=data_cfg["annot_path"]
    )

    train_subset = filter_by_ids(full_dataset, data_cfg["video_splits"]["train"])
    val_subset = filter_by_ids(full_dataset, data_cfg["video_splits"]["validation"])
    test_subset = filter_by_ids(full_dataset, data_cfg["video_splits"]["test"])

    train_dataset = PersonTransformSubset(train_subset, train_transform)
    val_dataset = PersonTransformSubset(val_subset,   val_transform)
    test_dataset = PersonTransformSubset(test_subset,  test_transform)

    loader_kwargs = {
        "batch_size": training_cfg["batch_size"],
        "num_workers": training_cfg.get("num_workers", 4),
        "pin_memory": training_cfg.get("pin_memory", True),
    }

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader