from torch.utils.data import Subset, Dataset
import torch

LABEL_TO_ID = {
    "l-pass": 0,
    "r-pass": 1,
    "l-spike": 2,
    "r-spike": 3,
    "l-set": 4,
    "r-set": 5,
    "l-winpoint": 6,
    "r-winpoint": 7,
}

ID_TO_PERSON_ACTION  = {
    0: 'waiting',
    1: 'setting',
    2: 'digging',
    3: 'falling',
    4: 'spiking',
    5: 'blocking',
    6: 'jumping',
    7: 'moving',
    8: 'standing'
}

PERSON_ACTION_TO_ID = {v: k for k, v in ID_TO_PERSON_ACTION .items()}

def normalize_activity_name(activity):
    return activity.strip().replace("_", "-")


def activity2id(activity):
    normalized = normalize_activity_name(activity)
    if normalized not in LABEL_TO_ID:
        raise KeyError(f"Unknown activity label: {activity}")
    return LABEL_TO_ID[normalized]


def id2activity(label_id):
    for action, current_id in LABEL_TO_ID.items():
        if current_id == label_id:
            return action
    return ""


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.subset[idx]

        if len(sample) == 3:
            # B1: (video_id, img, label)
            video_id, img, label = sample
            return self.transform(img), label

        else:
            # B4: (video_id, clip_id, imgs, label)
            video_id, clip_id, imgs, label = sample
            imgs_t = torch.stack([self.transform(img) for img in imgs])
            return imgs_t, label

    def __len__(self):
        return len(self.subset)

def filter_by_ids(dataset, ids):
    allowed_ids = {str(v) for v in ids}
    indices = [i for i, s in enumerate(dataset.samples) if s[0] in allowed_ids]
    return Subset(dataset, indices)