import torch
import os

from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torchmetrics import Accuracy, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.engine.utils import set_seed
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import get_logger
from src.utils.metrics import plot_training_curves


def run_epoch(model, loader, criterion, num_classes, device, optimizer=None, scaler=None):
    is_train = optimizer is not None


    eval_model = model.module if (not is_train and hasattr(model, 'module')) else model
    eval_model.train() if is_train else eval_model.eval()

    acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1_metric  = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    total_loss, total = 0.0, 0

    use_amp = device == 'cuda'

    with torch.enable_grad() if is_train else torch.no_grad():
        for *_, imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()
                with autocast('cuda', dtype=torch.float16, enabled=use_amp):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast('cuda', dtype=torch.float16, enabled=use_amp):
                    outputs = eval_model(imgs)
                    loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            acc_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            total_loss += loss.item() * labels.size(0)
            total      += labels.size(0)

    return total_loss / total, acc_metric.compute().item(), f1_metric.compute().item()


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_classes,
    num_epochs,
    model_name,
    path,
    scheduler=None,
    seed=42,
):
    log = get_logger(f"train_{model_name}", path.log("train"))
    log.info(f"Device: {device} | Seed: {seed}")
    set_seed(seed)

    scaler   = GradScaler('cuda', enabled=device == 'cuda')
    ckpt_mgr = CheckpointManager(save_path=path.checkpoints, keep_top_k=3, logger=log)
    history  = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "train_f1", "val_f1")}

    for epoch in range(1, num_epochs + 1):
        log.info(f"── Epoch [{epoch}/{num_epochs}] ───────────────")

        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, num_classes, device, optimizer=optimizer, scaler=scaler
        )

        torch.cuda.empty_cache()

        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, num_classes, device
        )

        if scheduler:
            scheduler.step(val_loss) if isinstance(scheduler, ReduceLROnPlateau) else scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        log.info(f"train loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f}")
        log.info(f"val loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")
        log.info(f"lr={lr:.2e}")

        for key, val in zip(history, (train_loss, val_loss, train_acc, val_acc, train_f1, val_f1)):
            history[key].append(val)

        ckpt_mgr.save(model, optimizer, epoch, val_acc)

        curves_path = path.figure("training_curves.png")
        plot_training_curves(history, curves_path)
        log.info(f"Curves : {curves_path}")

    return history