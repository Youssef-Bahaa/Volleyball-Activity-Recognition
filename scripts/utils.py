import argparse
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


def apply_cli_overrides(cfg: dict, args):
    """Merge any CLI args that were explicitly provided into the config dict."""
    overrides = {
        ("training", "epochs") : args.epochs,
        ("training", "learning_rate")          : args.lr,
        ("training", "optimizer") : args.optimizer,
        ("training", "scheduler") : args.scheduler,
        ("training", "criterion") : args.criterion,
        ("training", "weight_decay"): args.weight_decay,
        ("training", "seed") : args.seed,
        ("data", "batch_size")  : args.batch_size,
        ("data", "num_workers") : args.num_workers,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            cfg[section][key] = value
    return cfg





def parse_args():
    p = argparse.ArgumentParser(
        description= 'Train any registered model on the Volleyball Activity dataset',
        formatter_class= argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument('--model', required=True,
                   help= 'Model key in model_registry.py  e.g. B1, B3, B7')

    p.add_argument("--config", default=None,
                   help="Config YAML path. Defaults to configs/<MODEL>_config.yaml")

    p.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    p.add_argument("--lr", type=float, default=None, help="Override training.lr")
    p.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size")
    p.add_argument("--optimizer", type=str, default=None, help="Override training.optimizer")
    p.add_argument("--scheduler", type=str, default=None, help="Override training.scheduler")
    p.add_argument("--criterion", type=str, default=None, help="Override training.criterion")
    p.add_argument("--weight-decay",type=float, default=None, help="Override training.weight_decay")
    p.add_argument("--num-workers", type=int, default=None, help="Override data.num_workers")


    p.add_argument("--resume", default=None,
                   help="Checkpoint .pth to resume from")
    p.add_argument("--seed", type=int, default=None,
                   help="Override training.seed")

    return p.parse_args()



def build_optimizer(cfg: dict, model):
    lr = cfg['training']['learning_rate']
    return torch.optim.Adam(model.parameters(),lr=lr)



def build_scheduler(cfg: dict, optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=cfg['training']['scheduler_mode'], factor=cfg['training']['scheduler_factor'],
        patience=cfg['training']['scheduler_patience'], min_lr=cfg['training']['min_lr']
    )
    return scheduler


def build_criterion(cfg: dict):
    return nn.CrossEntropyLoss()





