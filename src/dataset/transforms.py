from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
    A.Resize(224, 224),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.ColorJitter(brightness=0.2),
        A.RandomBrightnessContrast(),
        A.GaussNoise()
    ], p=0.5),
    A.OneOf([
        A.HorizontalFlip(),
        A.VerticalFlip(),
    ], p=0.05),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

test_transform = val_transform