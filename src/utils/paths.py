import os
from pathlib import Path

def _is_kaggle():
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

class Paths:
    def __init__(self, root, model_name):
        self.model_name = model_name
        self.root = Path(root)

        # folders
        self.checkpoints = self.root / "checkpoints" / model_name
        self.results = self.root / "results" / model_name
        self.figures = self.results / "figures"
        self.logs = self.root / "logs" / model_name
        self.configs = self.root / "config"

        # ── data paths: Kaggle vs local ──────────────────────
        if _is_kaggle():
            kaggle_input = Path("/kaggle/input")
            # find the dataset folder automatically
            dataset_dirs = [d for d in kaggle_input.iterdir() if d.is_dir()]
            if not dataset_dirs:
                raise FileNotFoundError("No dataset found under /kaggle/input/")
            dataset_root = r'/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball'

            self.data = dataset_root
            self.videos = dataset_root / "videos"
            self.annot = dataset_root / "annot_all.pkl"
            self.tracking_annot = dataset_root / "volleyball_tracking_annotation"
        else:
            self.data = self.root / "data"
            self.videos = self.data / "videos_dataset"
            self.annot = self.root / "src" / "dataset" / "annot_all.pkl"
            self.tracking_annot = self.data / "volleyball_tracking_annotation"

        self.yolo_dataset = self.data / "yolo_dataset"
        self.yolo_runs = self.root / "checkpoints" / "YOLO" / "runs"

        self._create_dirs()

    # ── File Helpers ─────────────────────────────────────────

    def checkpoint(self, name):
        if not name.endswith(".pth"):
            name += ".pth"
        return self.checkpoints / name

    def figure(self, name):
        return self.figures / name

    def log(self, name):
        return self.logs / name

    def result(self, name):
        return self.results / name

    def config(self, name=None):
        name = name or f"{self.model_name}_config.yaml"
        return self.configs / name

    # ── Best checkpoint ──────────────────────────────────────

    def best_checkpoint(self):
        pth_files = [f for f in os.listdir(self.checkpoints) if f.endswith('.pth')]
        if not pth_files:
            raise FileNotFoundError(f"No .pth files in {self.checkpoints}")

        best_file = None
        best_acc  = -1

        for f in pth_files:
            try:
                acc = float(f.split("_acc_")[1].replace(".pth", ""))
                if acc > best_acc:
                    best_acc  = acc
                    best_file = f
            except:
                continue

        if best_file is None:
            raise ValueError("No valid checkpoint filenames with accuracy found")

        return self.checkpoints / best_file

    def last_checkpoint(self):
        last = self.checkpoints / "last.pth"
        if not last.exists():
            raise FileNotFoundError(f"No last.pth in {self.checkpoints}")
        return last

    def __repr__(self):
        return f'Paths(model={self.model_name}, root={self.root})'

    def _create_dirs(self):
        for d in [self.checkpoints, self.figures, self.logs, self.results]:
            d.mkdir(parents=True, exist_ok=True)