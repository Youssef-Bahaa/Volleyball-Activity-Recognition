import os
from pathlib import Path

class Paths:
    def __init__(self, root, model_name):
        self.model_name = model_name
        self.root = Path(root)

        # folders
        self.checkpoints = self.root / "checkpoints" / model_name
        self.results = self.root / "results" / model_name
        self.figures = self.results / "figures"
        self.logs = self.root / "logs" / model_name

        self.configs = self.root / "configs"
        self.data = self.root / "data"
        self.videos = self.data / "videos_dataset"
        self.annot = self.root / "src" / "dataset" / "annot_all.pkl"

        self._create_dirs()

    #____________File Helpers____________

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

    def config(self, name = None):
        name = name or f"{self.model_name}_config.yaml"
        return self.configs / name


    # ____________best checkpoint____________
    def best_checkpoint(self):
        pth_files = [f for f in os.listdir(self.checkpoints) if f.endswith('.pth')]
        if not pth_files:
            raise FileNotFoundError(f"No .pth files in {self.checkpoints}")

        #'ckpt_epoch_{epoch}_acc_{val_acc:.4f}.pth'
        best_file = None
        best_acc = -1

        for f in pth_files:
            try:
                acc_part = f.split("_acc_")[1].replace(".pth", "")
                acc = float(acc_part)

                if acc > best_acc:
                    best_acc = acc
                    best_file = f
            except:
                continue

        if best_file is None:
            raise ValueError("No valid checkpoint filenames with accuracy found")

        return self.checkpoints / best_file


    # ____________Internal____________

    def __repr__(self):
        return f'Paths(model={self.model_name}, root={self.root})'

    def _create_dirs(self):
        for d in [
            self.checkpoints,
            self.figures,
            self.logs,
            self.results
        ]:
            d.mkdir(parents=True, exist_ok=True)







