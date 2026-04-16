import torch
from tqdm import tqdm
from torchmetrics import Accuracy, F1Score

def run_epoch(model, loader, criterion, optimizer, num_classes, is_train, device='cpu'):
    """
    Runs a single train or val epoch.
    Returns (avg_loss, accuracy, macro_f1).
    """

    model.train() if is_train else model.eval()

    acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1_metric  = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    total_loss, total = 0, 0,
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    desc = "  train" if is_train else "  val  "

    with ctx:
        for _, imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            preds = outputs.argmax(dim=1)
            acc_metric.update(preds, labels)
            f1_metric.update(preds, labels)

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)


    avg_loss = total_loss / total
    accuracy = acc_metric.compute().item()
    macro_f1 = f1_metric.compute().item()

    return avg_loss, accuracy, macro_f1
