import torch.optim as optim
import torch.optim.lr_scheduler
import yaml
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def build_optimizer(cfg: dict, model):
    weight_decay = cfg['training'].get('weight_decay', 0.0)
    optimizer_name = cfg['training'].get('optimizer', 'Adam')

    base_model = model.module if hasattr(model, 'module') else model

    if 'lr_backbone' in cfg['training'] and hasattr(base_model, 'backbone'):
        lr_backbone = cfg['training']['lr_backbone']
        lr_head = cfg['training']['lr_head']
        param_groups = [
            {'params': [p for i, c in enumerate(base_model.backbone.children())
                        for p in c.parameters() if i >= 6], 'lr': lr_backbone},
            {'params': base_model.layer_norm.parameters(), 'lr': lr_head},
            {'params': base_model.lstm.parameters(), 'lr': lr_head},
            {'params': base_model.fc.parameters(), 'lr': lr_head},
        ]
    else:
        lr = cfg['training']['learning_rate']
        param_groups = [{'params': model.parameters(), 'lr': lr}]

    if optimizer_name == 'AdamW':
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")



def build_scheduler(cfg: dict, optimizer):
    sched_type = cfg['training']['scheduler']

    if sched_type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg['training']['scheduler_T_max'],
            eta_min=float(cfg['training']['min_lr'])
        )

    elif sched_type == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg['training']['scheduler_mode'],
            factor=cfg['training']['scheduler_factor'],
            patience=cfg['training']['scheduler_patience'],
            min_lr=float(cfg['training']['min_lr'])
        )

    elif sched_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg['training']['step_size'],
            gamma=cfg['training']['gamma']
        )

    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")






