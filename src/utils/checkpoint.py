import os
import torch

class CheckpointManager:
    """
    Saves checkpoints and keeps only the top-k by val_acc.
    Usage:
        ckpt_mgr = CheckpointManager(save_dir="checkpoints", keep_top_k=3)
        ckpt_mgr.save(model, optimizer, epoch=5, val_acc=0.82)
    """

    def __init__(self, save_path, keep_top_k = 3):
        pass

    def save(self, model, optimizer, epoch, val_acc):
        pass

    def best_path(self):
        pass

    def load(self, model, optimizer = None, device = 'cpu'):
        pass