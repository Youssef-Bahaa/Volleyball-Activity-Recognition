import argparse
import importlib
import torch
import torch.nn as nn

from src.utils.checkpoint import CheckpointManager
from src.utils.paths import Paths
from src.engine.utils import get_config
from src.utils import checkpoint
from src.engine.evaluator import evaluate



LOADER_REGISTRY = {
    "B1": "src.dataset.DataLoader.B1_loader",
    "B3_phase1": "src.dataset.DataLoader.B3_person_loader",
    "B3": "src.dataset.DataLoader.B3_features_loader",
    'B4': 'src.dataset.DataLoader.B4_loader',
    'B5': 'src.dataset.DataLoader.B5_PersonTemp',
    'B5_GROUP': 'src.dataset.DataLoader.B5_VolleyBallScene',
    'B6_Person': 'src.dataset.DataLoader.B6_Person',
    'B6': 'src.dataset.DataLoader.B6_features_loader',
    'B7_Person': 'src.dataset.DataLoader.B5_PersonTemp',
    'B8_GROUP': 'src.dataset.DataLoader.B6_Group',
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
        "phases": ["train"],
        "loader": "src.dataset.DataLoader.B3_person_loader",
        "loader_fn": "build_person_loaders",
    },

    # ── B3 Phase 2+3: extract features then train scene model ──
    "B3": {
        "module": "src.models.B3.B3_model",
        "class": "B3Model",
        "phases": ["extract", "train"],
        "loader": "src.dataset.DataLoader.B3_loader",
        "extractor_module": "src.models.B3.B3_extractor",
        "extractor_class": "B3Extractor",
        "extractor_ckpt": "checkpoints/B3_phase1",
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
    "B6": {
        "module": "src.models.B6.Group_Temporal",
        "class": "GroupActivityB6",
        "phases": ["train"],
        "loader": "src.dataset.DataLoader.B6_features_loader",
    },
    "B7_Person": {
        "module": "src.models.B7.Person_Temporal",
        "class": "PersonTemp",
        "phases": ["train"],
        "loader": "src.dataset.DataLoader.B5_PersonTemp",
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

def load_model(name, nclasses, cfg=None):
    entry = MODEL_REGISTRY[name]
    module_path = entry["module"]
    class_name = entry["class"]

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

    if name == "B5_GROUP":
        from src.models.B5.Person_Temporal import PersonTemp
        person_model = PersonTemp(num_classes=9)
        person_ckpt = Paths('.', model_name='B5').best_checkpoint()
        CheckpointManager.load(person_ckpt, person_model, device='cpu')
        person_model.eval()
        return cls(num_classes=nclasses, player_model=person_model)

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

    return cls(num_classes=nclasses)


def load_loaders(model_name, cfg):
    module = importlib.import_module(LOADER_REGISTRY[model_name])
    return module.build_loaders(cfg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_config(args.config)
    p = Paths('.', model_name=args.model)


    num_classes = cfg['model']['num_classes']
    model = load_model(args.model, nclasses=num_classes, cfg=cfg).to(device)
    _, _, test_loader = load_loaders(model_name=args.model, cfg=cfg)
    best_model_pth = p.best_checkpoint()
    CheckpointManager.load(best_model_pth, model, device=device)


    dct = evaluate(
        model,
        test_loader,
        num_classes,
        device,
        p
    )

if __name__ == "__main__":
    main()








