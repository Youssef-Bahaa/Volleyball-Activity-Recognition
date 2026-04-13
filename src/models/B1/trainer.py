import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader , random_split, Subset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset.DataLoader.B1_loader import VolleyBallFeaturesDataset
from src.models.B1.B1_model import ResNetFineTune


#__________Configs_________
VIDEOS_ROOT = 'data/videos'
ANNOT_PATH = 'src/dataloader/annot_all.pkl'
NUM_CLASSES = 8
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

train_ids = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
val_ids = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
test_ids = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]


#__________Transforms_________
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


#__________Data_________
def filter_by_ids(dataset, ids):
    indices = [i for i, s in enumerate(dataset.samples) if int(s[0]) in ids]
    return Subset(dataset, indices)

def build_loaders():
    train_data = VolleyBallFeaturesDataset(VIDEOS_ROOT,ANNOT_PATH,train_transform)
    val_data = VolleyBallFeaturesDataset(VIDEOS_ROOT, ANNOT_PATH, test_transform)

    train_data = filter_by_ids(train_data, train_ids)
    val_data = filter_by_ids(val_data, val_ids)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)

    return train_loader , val_loader


#__________Train / Eval_________
def run_epoch(model,loader,criterion,optimizer,is_train):
    model.train() if is_train else model.eval()

    total_loss , correct , total = 0,0,0

    ctx = torch.enable_grad() if is_train else torch.no_grad

    with ctx:
        for _,img,label in tqdm(loader,desc='train' if is_train else 'val' , leave = False):
            imgs , labels = imgs.to(DEVICE) , labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(labels,outputs)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * img.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += img.size(0)

    return total_loss / total , correct / total



#__________Plotting_________
def plot_metrics(history , save_path):
    epochs = range(1 , NUM_EPOCHS + 1)
    fig, (ax1,ax2) = plt.subplots(1 , 2 , figsize = (12,4))

    ax1.plot(epochs , history['train_loss'] , label = 'Train')
    ax1.plot(epochs, history['val_loss'], label='Val')
    ax1.set_title('Loss')
    ax1.xlabel('Epoch')
    ax1.legend()

    ax2.plot(epochs , history['train_acc'] , label = 'Train')
    ax2.plot(epochs, history['val_acc'], label='Val')
    ax2.set_title('Accuracy')
    ax2.xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'Metrics plots saved to {save_path}')




#__________Main_________
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_loader, val_loader= build_loaders()
    print(f"Dataset -> train: {len(train_loader.dataset)}  |  val: {len(val_loader.dataset)}")
    print(f"Device  ->  {DEVICE}\n")

    model = ResNetFineTune(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),LR,)
    schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer , T_max=NUM_EPOCHS, eta_min=1e-6)

    history = {'train_loss': [] , 'val_loss': [] , 'val_loss': [] , 'test_loss': []}

    best_val_acc = 0.0

    for epoch in range(1 , NUM_EPOCHS + 1):
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]")

        train_loss,train_acc = run_epoch(model,train_loader,criterion,optimizer,True)
        val_loss,val_acc = run_epoch(model, val_loader, criterion, optimizer, False)


        schedular.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"train loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"val   loss: {val_loss:.4f}    acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            ckpt_path = os.path.join(SAVE_DIR, "best_model.pth")
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
            }, ckpt_path)

            print(f"New best checkpoint saved  (val_acc={val_acc:.4f})")

    plot_metrics(history, os.path.join(SAVE_DIR, "training_curves.png"))
    print(f"\nTraining complete.  Best val accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train()