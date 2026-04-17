import os
import torch
from src.utils.logger import get_logger

logger = get_logger("checkpoint", "logs/app.log")

class CheckpointManager:
    """
    Saves checkpoints and keeps only the top-k by val_acc.
    Usage:
        ckpt_mgr = CheckpointManager(save_dir="checkpoints", keep_top_k=3)
        ckpt_mgr.save(model, optimizer, epoch=5, val_acc=0.82)
    """

    def __init__(self, save_path, keep_top_k = 3):
        self.save_path  = save_path
        self.keep_top_k = keep_top_k
        self.history = []  # List of (val_acc, path)

    def save(self, model, optimizer, epoch, val_acc):
        filename = f"epoch_{epoch:02d}_acc{val_acc:.4f}.pth"
        path = os.path.join(self.save_dir, filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc
        }, path)

        logger.info(f"Saved checkpoint: {path}")

        self.history.append((val_acc, path))
        self.history.sort(key=lambda x: x[0] , reverse=True)  # Sort by val_acc descending

        #Remove old checkpoints
        for _, old_path in self.history[self.keep_top_k:]:
            if os.path.exists(old_path):
                os.remove(old_path)
                logger.info(f'Removed old checkpoint: {old_path}')

        self.history = self.history[:self.keep_top_k]
        logger.info(f"Checkpoint saved → {path}")
        return path

    def best_path(self):
        """Returns the path of the best checkpoint saved so far."""
        return self.history[0][1] if self.history else None

    def load(self, path, model, optimizer = None, device = 'cpu'):
        """
        Loads a checkpoint into model (and optionally optimizer).
        Returns the checkpoint dict for epoch/val_acc inspection.
        """
        ckpt = torch.load(path, map_location= device)
        state = ckpt.get('model_state_dict' , ckpt)
        model.load(state)

        if optimizer is not None and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])


        logger.info(
            f'Loaded Checkpoint: {path}' +
             f"(epoch={ckpt.get('epoch')}, val_acc={ckpt.get('val_acc', 'n/a')}"
        )

        return ckpt


