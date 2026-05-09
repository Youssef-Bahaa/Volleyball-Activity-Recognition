import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from src.dataset.utils import activity2id, filter_by_ids
import pickle
from src.utils.paths import Paths

import src.dataset.boxinfo as boxinfo_module
import sys

sys.modules['boxinfo'] = boxinfo_module

class FeaturesDataset(Dataset):
    """Loads pre-extracted per-clip .npy features from disk."""

    def __init__(self, root_path, annot):
        self.samples = []

        for video_id in os.listdir(root_path):
            clips_dir = os.path.join(root_path, video_id)
            if not os.path.isdir(clips_dir):
                continue

            for clip_file in os.listdir(clips_dir):
                if not clip_file.endswith(".npy"):
                    continue
                clip_id = clip_file.replace(".npy", "")
                feature_path = os.path.join(clips_dir, clip_file)
                activity = annot[video_id][clip_id]["category"]
                self.samples.append((video_id, feature_path, activity))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, feature_path, activity = self.samples[idx]
        features = torch.tensor(np.load(feature_path), dtype=torch.float32)  # (N, 2048)
        return features, activity2id(activity)


def build_loaders(cfg):

    p = Paths('.', model_name='B6_extract')
    feat_dir = str(p.result("features_resnet"))

    with open(cfg["data"]["annot_path"], "rb") as f:
        annot = pickle.load(f)

    splits  = cfg["data"]["video_splits"]
    dataset = FeaturesDataset(feat_dir, annot)

    def filter_by_ids(ids):
        allowed = {str(i) for i in ids}
        indices = [i for i, s in enumerate(dataset.samples) if s[0] in allowed]
        return Subset(dataset, indices)

    kw = {
        "batch_size": cfg["training"]["batch_size"],
        "num_workers": cfg["training"].get("num_workers", 2),
        "pin_memory": cfg["training"].get("pin_memory", True),
    }

    return (
        DataLoader(filter_by_ids(splits["train"]), shuffle=True, **kw),
        DataLoader(filter_by_ids(splits["validation"]), shuffle=False, **kw),
        DataLoader(filter_by_ids(splits["test"]), shuffle=False, **kw),
    )