import torch
import numpy as np
from tqdm import tqdm
from torchmetrics import Accuracy, F1Score
from src.utils.logger import get_logger

logger = get_logger("evaluator", "logs/app.log")


def evaluate(model, loader, criterion, num_classes, device='cpu'):
    """
    Full evaluation pass.
    Returns (avg_loss, accuracy, macro_f1, all_labels, all_preds).
    """
    logger.debug(f"Starting evaluation on device: {device}")
    model.eval()

    acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1_metric  = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)


    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for _, imgs, labels in tqdm(loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            acc_metric.update(preds, labels)
            f1_metric.update(preds, labels)

    avg_loss = total_loss / total
    accuracy = acc_metric.compute().item()
    macro_f1 = f1_metric.compute().item()

    logger.info(f"Evaluation complete - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}")

    return avg_loss, accuracy, macro_f1, np.array(all_labels), np.array(all_preds)