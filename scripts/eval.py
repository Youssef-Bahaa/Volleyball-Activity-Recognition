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
    "B2": "src.dataset.DataLoader.B2_loader",
}

MODEL_REGISTRY = {
    "B1": ("src.models.B1.B1_model", "ResNetFineTune"),
    "B2": ("src.models.B2.B2_model", "."),

}

def load_model(name, nclasses, pretrained= True):
    module_path, class_name = MODEL_REGISTRY[name]
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(num_classes=nclasses, pretrained=pretrained)


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
    model = load_model(args.model, nclasses =num_classes,pretrained= cfg['model']['pretrained']).to(device)
    _, _, test_loader = load_loaders(model_name=args.model, cfg=cfg)
    best_model_pth = p.best_checkpoint()
    CheckpointManager.load(best_model_pth, model, device=device)


    evaluate(
        model,
        test_loader,
        num_classes,
        device,
        p
    )


if __name__ == "__main__":
    main()








