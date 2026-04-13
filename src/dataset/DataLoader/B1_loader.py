import os
from torch.utils.data import Dataset
import pickle
from PIL import Image
import torch


class VolleyBallFeaturesDataset(Dataset):
    def __init__(self ,image_root, annot_path, transform):
        self.samples = []
        self.transform = transform
        videos_path = image_root

        with open(annot_path, 'rb') as f:
            data = pickle.load(f)

        for video_id in os.listdir(videos_path):
            video_dir = os.path.join(videos_path ,video_id)
            if not os.path.isdir(video_dir):
                continue

            for clip_file in os.listdir(video_dir):
                file_path = os.path.join(video_dir ,clip_file)
                if not os.path.isdir(file_path):
                    continue

                img_path = os.path.join(file_path ,f'{clip_file}.jpg') # getting the middle image

                activity  = data[video_id][clip_file]['category']
                activity = activity2id(activity)
                self.samples.append((video_id ,img_path ,activity))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, img_pth, label = self.samples[idx]
        img = Image.open(img_pth).convert('RGB')
        img = self.transform(img)

        return video_id, img, torch.tensor(label, dtype=torch.long)

