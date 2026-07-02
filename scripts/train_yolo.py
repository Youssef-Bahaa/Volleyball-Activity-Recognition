import argparse
import os
import shutil
import sys
from ultralytics import YOLO
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.paths import Paths

def get_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = argparse.ArgumentParser
    args.add_argument('config', default="config/YOLO.yaml")
    args.add_argument('resume', action="store_true", default=False)

    cfg = get_config(args.config)

    p = Paths(".", model_name="YOLO")
    data_yaml = os.path.join(cfg["dataset"]["output_path"], "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"{data_yaml} not found — run scripts/prepare_yolo_dataset.py first.")

    train_cfg = cfg["training"]
    exp_name = cfg["experiment"]["name"]

    model = YOLO(cfg["model"]["base_weights"])

    model.train(
        data=data_yaml,
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        patience=train_cfg.get("patience", 15),
        optimizer=train_cfg.get("optimizer", "AdamW"),
        lr0=train_cfg.get("lr0", 0.001),
        device=train_cfg.get("device", 0),
        workers=train_cfg.get("workers", 4),
        seed=train_cfg.get("seed", 42),
        freeze=train_cfg.get("freeze", 0) or None,
        project=str(p.yolo_runs),
        name=exp_name,
        resume=args.resume,
        exist_ok=True,
    )

    run_dir = os.path.join(str(p.yolo_runs), exp_name, "weights")
    best_src = os.path.join(run_dir, "best.pt")

    if os.path.exists(best_src):
        best_dst = p.checkpoints / "best.pt"
        shutil.copy2(best_src, best_dst)
        print(f"Best weights copied to {best_dst}")

    metrics = model.val(data=data_yaml, imgsz=train_cfg["imgsz"], split="test")
    print("Test metrics:", metrics.results_dict)

if __name__ == "__main__":
    main()

