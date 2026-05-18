import torch
from tqdm import tqdm
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

from src.dataset.utils import LABEL_TO_ID
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import get_logger
from src.utils.metrics import plot_training_curves,plot_confusion_matrix,plot_per_class_accuracy,save_classification_report

def evaluate(model, test_loader, num_classes, device, path):
    log = get_logger("evaluator", path.log("eval"))
    log.info("Starting evaluation...")

    model.eval()

    acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1_metric  = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    cm_metric  = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    all_preds = []
    all_labels = []


    with torch.no_grad():
        for *_, imgs, labels in tqdm(test_loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            acc_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            cm_metric.update(preds, labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = acc_metric.compute().item()
    f1  = f1_metric.compute().item()
    cm  = cm_metric.compute()

    cm_path = path.figure("confusion_matrix.png")
    acc_path = path.figure("per_class_acc.png")
    report_path = path.figure("classification_report.txt")

    class_names = list(LABEL_TO_ID.keys())

    plot_confusion_matrix(all_labels, all_preds, class_names, cm_path)
    plot_per_class_accuracy(all_labels, all_preds, class_names, acc_path)
    save_classification_report(all_labels, all_preds, loss=0.0, save_path=report_path, acc=acc)

    log.info(f"Test  acc={acc:.4f}  f1={f1:.4f}")
    log.info(f"Confusion Matrix:\n{cm}")

    return {"acc": acc, "f1": f1, "confusion_matrix": cm}