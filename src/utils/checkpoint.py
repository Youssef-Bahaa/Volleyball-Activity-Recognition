import os
import torch
from src.utils.logger import get_logger

#logger = get_logger("checkpoint", "logs/app.log")

class CheckpointManager:
    """
    Saves checkpoints and keeps only the top-k by val_acc.
    Usage:
        ckpt_mgr = CheckpointManager(save_dir="checkpoints", keep_top_k=3)
        ckpt_mgr.save(model, optimizer, epoch=5, val_acc=0.82)
    """

    def __init__(self, save_path, keep_top_k = 1, logger=None):
        self.save_path  = save_path
        self.keep_top_k = keep_top_k
        self.history = []  # List of (val_acc, path)
        self.logger = logger
        os.makedirs(save_path, exist_ok=True)

    def save(self, model, optimizer, epoch, val_acc, early_stopping=None):
        filename = f"epoch_{epoch:02d}_acc_{val_acc:.4f}.pth"
        path = os.path.join(self.save_path, filename)

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc}

        if early_stopping is not None:
            state["early_stopping"] = early_stopping.state_dict()

        torch.save(state, path)
        self.logger.info(f"Saved checkpoint: {path}")

        # ── always save last epoch separately ───────────────────
        last_path = os.path.join(self.save_path, "last.pth")
        torch.save(state, last_path)
        self.logger.info(f"Saved last checkpoint: {last_path}")

        self.history.append((val_acc, path))
        self.history.sort(key=lambda x: x[0] , reverse=True)  # Sort by val_acc descending

        #Remove old checkpoints
        for _, old_path in self.history[self.keep_top_k:]:
            if os.path.exists(old_path):
                os.remove(old_path)
                self.logger.info(f'Removed old checkpoint: {old_path}')

        self.history = self.history[:self.keep_top_k]

        return path

    def best_path(self):
        """Returns the path of the best checkpoint saved so far."""
        return self.history[0][1] if self.history else None

    @staticmethod
    def load(path, model, optimizer = None, device = 'cpu', early_stopping=None,logger=None):
        """
        Loads a checkpoint into model (and optionally optimizer).
        Returns the checkpoint dict for epoch/val_acc inspection.
        """
        ckpt = torch.load(path, map_location= device)
        state = ckpt.get('model_state_dict' , ckpt)
        model.load_state_dict(state)

        if optimizer is not None and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if early_stopping is not None and "early_stopping" in ckpt:
            early_stopping.load_state_dict(ckpt["early_stopping"])

        if logger:
            logger.info(f"Loaded {path} (epoch={ckpt.get('epoch')}, val_acc={ckpt.get('val_acc', 'n/a')})")

        return ckpt


