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


class VolleyBallB6GroupDataset(Dataset):
    def __init__(self, image_root, annot_path, num_persons=12):
        self.samples = []
        self.num_persons = num_persons

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
                    label = activity2id(clip_info['category'])
                    frame_boxes = clip_info['frame_boxes_dct']
                except KeyError:
                    continue

                sorted_frames = sorted(frame_boxes.keys())
                img_paths = [
                    os.path.join(clip_dir, f'{fid}.jpg')
                    for fid in sorted_frames
                    if os.path.exists(os.path.join(clip_dir, f'{fid}.jpg'))
                ]

                if not img_paths:
                    continue

                self.samples.append((video_id, clip_id, img_paths, sorted_frames, frame_boxes, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, clip_id, img_paths, sorted_frames, frame_boxes, label = self.samples[idx]

        persons_data = [[] for _ in range(self.num_persons)]

        for t, img_path in enumerate(img_paths):
            fid = sorted_frames[t]
            boxes = frame_boxes[fid]
            img = Image.open(img_path).convert('RGB')

            # sort players by x-center (left -> right across court = team split)
            boxes_sorted = sorted(boxes, key=lambda b: (b.box[0] + b.box[2]) / 2)

            for pid, box_info in enumerate(boxes_sorted):
                if pid >= self.num_persons:
                    break
                x1, y1, x2, y2 = box_info.box
                persons_data[pid].append(img.crop((x1, y1, x2, y2)))

            # pad missing players with None
            for pid in range(len(boxes_sorted), self.num_persons):
                persons_data[pid].append(None)

        return video_id, clip_id, persons_data, torch.tensor(label, dtype=torch.long)



class B6GroupTransformSubset(Dataset):
    """
    Applies transform to every player crop.
    None placeholders (missing players) become zero tensors.
    Output: (N, T, C, H, W), label
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        video_id, clip_id, persons_data, label = self.subset[idx]

        transformed = []
        for crops in persons_data: # iterate N players
            frames = []
            for crop in crops:  # iterate T frames
                if crop is None:
                    t = self.transform(Image.new('RGB', (224, 224)))
                    frames.append(torch.zeros_like(t))
                else:
                    frames.append(self.transform(crop))
            transformed.append(torch.stack(frames)) # (T, C, H, W)

        video_tensor = torch.stack(transformed)   # (N, T, C, H, W)
        return video_tensor, label


def build_loaders(cfg):
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]

    full_dataset = VolleyBallB6GroupDataset(
        image_root=data_cfg["videos_path"],
        annot_path=data_cfg["annot_path"]
    )

    train_subset = filter_by_ids(full_dataset, data_cfg["video_splits"]["train"])
    val_subset = filter_by_ids(full_dataset, data_cfg["video_splits"]["validation"])
    test_subset = filter_by_ids(full_dataset, data_cfg["video_splits"]["test"])

    train_dataset = B6GroupTransformSubset(train_subset, train_transform)
    val_dataset = B6GroupTransformSubset(val_subset, val_transform)
    test_dataset = B6GroupTransformSubset(test_subset, test_transform)

    loader_kwargs = {
        "batch_size": training_cfg["batch_size"],
        "num_workers": training_cfg.get("num_workers", 4),
        "pin_memory": training_cfg.get("pin_memory", True),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader