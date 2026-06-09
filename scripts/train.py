import argparse
import importlib
import torch
import torch.nn as nn

from src.utils.checkpoint import CheckpointManager
from src.engine.trainer import train
from src.utils.paths import Paths
from src.engine.utils import get_config, set_seed, build_optimizer, build_scheduler


LOADER_REGISTRY = {
    "B1": "src.dataset.DataLoader.B1_loader",
    "B3_phase1": "src.dataset.DataLoader.B3_person_loader",
    "B3": "src.dataset.DataLoader.B3_features_loader",
    'B4': 'src.dataset.DataLoader.B4_loader',
    'B5': 'src.dataset.DataLoader.B5_PersonTemp',
    'B5_GROUP': 'src.dataset.DataLoader.B5_VolleyBallScene',
    'B6_Person': 'src.dataset.DataLoader.B6_Person',
    'B6': 'src.dataset.DataLoader.B6_features_loader',
    'B7_Person': 'src.dataset.DataLoader.B7_Person',
    'B7_Group': 'src.dataset.DataLoader.B6_Group',
    'B8_Group': 'src.dataset.DataLoader.B6_Group',
}

MODEL_REGISTRY = {
    # ── Single phase ───────────────────────────────────────────
    "B1": {
        "module": "src.models.B1.B1_model",
        "class": "ResNetFineTune",
        "phases": ["train"],
        "loader": "src.dataset.DataLoader.B1_loader",
    },

    # ── B3 Phase 1: train person-action classifier ─────────────
    "B3_phase1": {
        "module": "src.models.B3.B3_extractor",
        "class": "B3Extractor",
        "phases": ["train", "extract"],
        "loader": "src.dataset.DataLoader.B3_person_loader",
        "extractor_module": "src.models.B3.extractor",
        "extractor_class": "B3Extractor",
    },

    # ── B3 train scene model ──
    "B3": {
        "module": "src.models.B3.B3_model",
        "class": "B3Model",
        "phases": ["train"],
        "loader": "src.dataset.DataLoader.B3_loader",
    },
    "B4": {
        "module": "src.models.B4.B4_model",
        "class": "B4Model",
        "phases": ["train"],
        "loader": "src.dataset.DataLoader.B4_loader",
    },
    "B5": {
        "module": "src.models.B5.Person_Temporal",
        "class": "PersonTemp",
        "phases": ["train"],
        "loader": "src.dataset.DataLoader.B5_PersonTemp",
    },
    "B5_GROUP": {
        "module": "src.models.B5.Group_Temporal",
        "class": "GroupActivityB5",
        "phases": ["train"],
        "loader": "src.dataset.DataLoader.B5_VolleyBallScene",
    },

    "B6_Person": {
        "module": "src.models.B6.Person",
        "class": "PersonClassifier",
        "phases": ["train", "extract"],
        "loader": "src.dataset.DataLoader.B6_Person",
    },
    "B6_Group": {
        "module": "src.models.B6.Group_Temporal",
        "class": "GroupActivityB6",
        "phases": ["train"],
        "loader": "src.dataset.DataLoader.B6_features_loader",
    },
    "B7_Person": {
        "module": "src.models.B7.Person_Temporal",
        "class": "PersonTemp",
        "phases": ["train"],
        "loader": "src.dataset.DataLoader.B7_Person",
    },
    "B7_Group": {
        "module": "src.models.B7.Group_Temporal",
        "class": "GroupActivityB7",
        "phases": ["train"],
        "loader": "src.dataset.DataLoader.B6_Group",
    },
    "B8_Group": {
        "module": "src.models.B8.Group_Temporal",
        "class": "GroupActivityB8",
        "phases": ["train"],
        "loader": "src.dataset.DataLoader.B6_Group",
    },
}
def _import(module_path, class_name):
    return getattr(importlib.import_module(module_path), class_name)

def load_extractor(name, num_classes, pretrained=True):
    info = MODEL_REGISTRY[name]
    cls  = _import(info["module"], info["class"])      # ← always use "module"/"class"
    return cls(num_classes=num_classes, pretrained=pretrained)

def load_model(name, nclasses, pretrained=True, cfg=None):
    model_info = MODEL_REGISTRY[name]

    module_path = model_info["module"]
    class_name = model_info["class"]

    cls = getattr(importlib.import_module(module_path), class_name)
    if name == "B4":
        b1_path = Paths('.', model_name='B1').best_checkpoint()
        return cls(
            b1_path=b1_path,
            input_dim=cfg['model']['input_dim'],
            hidden_dim=cfg['model']['hidden_dim'],
            num_layers=cfg['model']['num_layers'],
            num_classes=cfg['model']['num_classes'],
        )
    elif name == 'B5_GROUP':
        # load the person model (stage 1)
        person_cls = getattr(
            importlib.import_module("src.models.B5.Person_Temporal"), "PersonTemp"
        )
        person_model = person_cls()
        ckpt_path = Paths('.', model_name='B5').best_checkpoint()
        CheckpointManager.load(ckpt_path, person_model, device='cpu')
        return cls(player_model=person_model, num_classes=nclasses)

    elif name == 'B7_Group':
        person_cls = getattr(
            importlib.import_module("src.models.B7.Person_Temporal"), "PersonTemp"
        )
        person_model = person_cls()
        ckpt_path = Paths('.', model_name='B7_Person').best_checkpoint()
        CheckpointManager.load(ckpt_path, person_model, device='cpu')
        return cls(player_model=person_model, num_classes=nclasses)

    elif name == 'B8_Group':
        person_cls = getattr(
            importlib.import_module("src.models.B7.Person_Temporal"), "PersonTemp"
        )
        person_model = person_cls()
        ckpt_path = Paths('.', model_name='B7_Person').best_checkpoint()
        CheckpointManager.load(ckpt_path, person_model, device='cpu')
        return cls(player_model=person_model, num_classes=nclasses)

    return cls()


def load_loaders(model_name, cfg):
    module = importlib.import_module(LOADER_REGISTRY[model_name])
    return module.build_loaders(cfg)


def run_extract(args, cfg, p, device):
    """Phase 1 — extract and save features to disk."""
    num_classes = cfg['model']['num_classes']
    extractor = load_extractor(args.model, num_classes, pretrained=True).to(device)

    # load best checkpoint if one exists
    try:
        CheckpointManager.load(p.best_checkpoint(), extractor, device=device)
    except FileNotFoundError:
        pass  # no checkpoint yet — extract with pretrained weights

    module = importlib.import_module(MODEL_REGISTRY[args.model]["extractor_module"])
    module.extract_and_save(extractor, device, p, cfg)


def run_train(args, cfg, p, device):
    """Phase 2 — train the main model (on raw imgs or pre-extracted features)."""
    num_classes = cfg['model']['num_classes']
    model = load_model(args.model, num_classes, pretrained=cfg['model']['pretrained'], cfg=cfg).to(device)
    # GPU parallelism
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # Resume from checkpoint
    start_epoch = 1
    saved_history = {}
    if args.resume:
        try:
            ckpt = CheckpointManager.load(p.last_checkpoint(), model, optimizer, device=device)
            start_epoch = ckpt.get('epoch', 0) + 1
            saved_history = ckpt.get('history', {})
            print(f"Resumed from epoch {start_epoch - 1}")
            if scheduler is not None and 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict']:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                print(f"Scheduler state restored")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch")

    train_loader, val_loader, _ = load_loaders(args.model, cfg)

    patience = (
        args.patience if args.patience is not None else cfg['training'].get('patience', 7)
    ) if args.early_stopping else None

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(label_smoothing=cfg['training'].get('label_smoothing', 0.1)),
        optimizer=optimizer,
        device=device,
        num_classes=num_classes,
        num_epochs=cfg['training']['epochs'],
        model_name=args.model,
        path=p,
        scheduler=scheduler,
        seed=cfg['experiment']['seed'],
        start_epoch=start_epoch,
        patience=patience,
        cfg=cfg,
        saved_history=saved_history,
    )




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase",  choices=["extract", "train", "both"], default="train")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--early-stopping", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=None)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_config(args.config)
    p = Paths('.', model_name=args.model)


    model_phases = MODEL_REGISTRY[args.model]["phases"]
    if args.phase == "extract":
        assert "extract" in model_phases, f"{args.model} has no extract phase"
        run_extract(args, cfg, p, device)

    elif args.phase == "train":
        run_train(args, cfg, p, device)

    elif args.phase == "both":
        assert "extract" in model_phases, f"{args.model} has no extract phase"
        run_extract(args, cfg, p, device)
        run_train(args, cfg, p, device)



if __name__ == "__main__":
    main()








