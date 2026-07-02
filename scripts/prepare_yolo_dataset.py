import argparse
import os
import shutil
import sys
from collections import defaultdict

import yaml
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.boxinfo import BoxInfo


def get_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_clip_boxes(annot_txt_path):
    frame_boxes = defaultdict(list)
    with open(annot_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            box = BoxInfo(line)
            frame_boxes[box.frame_ID].append(box)
    return frame_boxes


def box_to_yolo_line(box, img_w, img_h):
    x1, y1, x2, y2 = box.box
    x1, x2 = sorted((max(0, min(x1, img_w)), max(0, min(x2, img_w))))
    y1, y2 = sorted((max(0, min(y1, img_h)), max(0, min(y2, img_h))))

    w = x2 - x1
    h = y2 - y1
    if w <= 1 or h <= 1:
        return None

    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    w_n = w / img_w
    h_n = h / img_h
    return f"0 {xc:.6f} {yc:.6f} {w_n:.6f} {h_n:.6f}"

def split_for_video(video_id, video_splits):
    vid = int(video_id)

    for split_name in ("train", "val", "test"):
        if vid in video_splits.get(split_name, []):
            return split_name

    return None


def place_file(src_path, dst_path, copy_images):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(src_path, dst_path)
    else:
        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()
        os.symlink(os.path.abspath(src_path), dst_path)

def build(cfg):
    ds_cfg = cfg["dataset"]
    videos_root = Path(ds_cfg["videos_path"])
    annot_root = Path(ds_cfg["tracking_annot_path"])
    out_root = Path(ds_cfg["output_path"])
    only_keyframe = ds_cfg.get("only_keyframe", False)
    include_generated = ds_cfg.get("include_generated", True)
    skip_lost = ds_cfg.get("skip_lost", True)
    copy_images = ds_cfg.get("copy_images", True)
    video_splits = ds_cfg["video_splits"]

    if not videos_root.exists():
        raise FileNotFoundError(f"videos_path not found: {videos_root}")
    if not annot_root.exists():
        raise FileNotFoundError(f"tracking_annot_path not found: {annot_root}")

    counts = defaultdict(int)
    skipped_no_split = 0

    video_ids = sorted(
        [d.name for d in videos_root.iterdir() if d.is_dir()],
        key=lambda x: int(x) if x.isdigit() else x,
    )

    for video_id in video_ids:
        split = split_for_video(video_id, video_splits)
        if split is None:
            skipped_no_split += 1
            continue

        video_dir = videos_root / video_id
        clip_ids = sorted([d.name for d in video_dir.iterdir() if d.is_dir()])

        for clip_id in clip_ids:
            annot_txt = annot_root / video_id / clip_id / f"{clip_id}.txt"
            if not annot_txt.exists():
                continue

            frame_boxes = load_clip_boxes(annot_txt)
            clip_dir = video_dir / clip_id

            if only_keyframe:
                frame_ids = [int(clip_id)] if int(clip_id) in frame_boxes else []
            else:
                frame_ids = sorted(frame_boxes.keys())

            for frame_id in frame_ids:
                boxes = frame_boxes.get(frame_id, [])
                if not boxes:
                    continue

                img_path = clip_dir / f"{frame_id}.jpg"
                if not img_path.exists():
                    continue

                kept_lines = []
                with Image.open(img_path) as im:
                    img_w, img_h = im.size

                for box in boxes:
                    if skip_lost and getattr(box, "lost", 0) == 1:
                        continue
                    if not include_generated and getattr(box, "generated", 0) == 1:
                        continue
                    line = box_to_yolo_line(box, img_w, img_h)
                    if line is not None:
                        kept_lines.append(line)

                if not kept_lines:
                    continue

                stem = f"{video_id}_{clip_id}_{frame_id}"
                dst_img = out_root / "images" / split / f"{stem}.jpg"
                dst_lbl = out_root / "labels" / split / f"{stem}.txt"

                place_file(img_path, dst_img, copy_images)
                with open(dst_lbl, "w") as f:
                    f.write("\n".join(kept_lines) + "\n")

                counts[split] += 1

        print(f"[{video_id}] processed (split={split}) : running totals {dict(counts)}")

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": cfg["model"]["num_classes"],
        "names": cfg["model"]["class_names"],
    }
    with open(out_root / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print("\nDone.")
    print(f"images written : {dict(counts)}")
    print(f"videos skipped (not in any split): {skipped_no_split}")
    print(f"data.yaml : {out_root / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/YOLO.yaml")
    args = parser.parse_args()

    cfg = get_config(args.config)
    build(cfg)

if __name__ == '__main__':
    main()

