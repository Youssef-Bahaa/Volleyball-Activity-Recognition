from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.2, hue=0.05)
    ], p=0.7),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomHorizontalFlip(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_transform = val_transform