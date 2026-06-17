# Volleyball Activity Recognition

A modern PyTorch reimplementation of the CVPR 2016 paper **"A Hierarchical Deep Temporal Model for Group Activity Recognition"** (Mostafa Saad Ibrahim, 2016).

> **Paper**: [Mostafa Saad Ibrahim, CVPR 2016](http://www.cs.sfu.ca/~mori/research/papers/ibrahim-cvpr16.pdf)  
> **Original Repository**: [mostafa-saad/deep-activity-rec](https://github.com/mostafa-saad/deep-activity-rec)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Baselines](#baselines)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Training](#training)
- [Configuration](#configuration)
- [Citation](#citation)

---

## Overview

Group activity recognition is the task of classifying the collective behavior of multiple people in a video clip.

This repository faithfully reimplements the hierarchical two-stage temporal model proposed by Mostafa Saad Ibrahim using modern PyTorch. The original work was built on Caffe-LSTM; this implementation replaces that stack entirely with a clean, reproducible PyTorch pipeline while staying true to the architectural ideas of the paper.

Key difference from the original:

- ResNet50 backbone (pretrained on ImageNet) replaces AlexNet/VGG

## Architecture

The model follows the two-stage hierarchical design described in the paper.

### Stage 1 — Person-Level Model (B8 Person)

Each annotated player in a clip is cropped to a 224x224 bounding box and passed through a fine-tuned ResNet50 backbone. The resulting feature vectors across T frames form a temporal sequence, which is fed into a single-layer LSTM to model that individual's action dynamics over time. The final hidden state represents the player's action embedding.

### Stage 2 — Group-Level Model (B8 Group)

All per-player embeddings from Stage 1 are pooled and arranged into a scene-level sequence. A second LSTM ingests this sequence to capture how the team's collective state evolves over time. The final hidden state is passed through a linear classifier to predict one of eight group activity labels.

```
Input Clip (T frames, K players)
        |
        v
  ResNet50 Backbone
  (per player, per frame)
        |
        v
  Person LSTM  (Stage 1)
  -> player action embedding
        |
        v
  Pooling over K players
        |
        v
  Group LSTM   (Stage 2)
  -> scene-level embedding
        |
        v
  Linear Classifier
  -> Group Activity Label
```

---

## Baselines

| Baseline | Description |
|----------|--------------|
| **B1** | Image Classification — a CNN fine-tuned on the full frame to directly predict group activity, with no per-person modeling. |
| **B3** | Fine-tuned Person Classification — a CNN fine-tuned on individual player crops to predict person-level actions, then pooled across players to classify group activity, with no temporal modeling. |
| **B4** | Temporal Model with Image Features — a CNN deployed on the full frame, with features fed into an LSTM to classify group activity over time. |
| **B5** | Temporal Model with Person Features — per-player CNN features pooled across players at each timestep and fed into an LSTM to classify group activity. |
| **B7** | Two-stage Model without Group LSTM — the full person-level temporal model (CNN + Person LSTM), with classification done directly on pooled person-level temporal features, omitting the second group-level LSTM. |
| **B8** | Full Two-stage Hierarchical Model — the complete architecture: Person LSTM (Stage 1) followed by Group LSTM (Stage 2). Not yet trained in this repository. |

### Baseline Results (this implementation)

| Baseline | Accuracy | F1-Score |
|----|----------|----------|
| B1 | 80.10%   | 81.05%   |
| B3 | 80.25%   | 74.86%   |
| B4 | 81.82%   | 81.87%   |
| B5 | 76.14%   | 71.87%   |
| B7 | **83.10%** | **82.92%** |

---

## Dataset

| Property              | Value   |
|-----------------------|---------|
| Total videos          | 55      |
| Annotated frames      | 4,830   |
| Frames per clip       | 41     |
| Frames used (this work)| 9      |
| Training frames       | 3,493   |
| Testing frames        | 1,337   |
| Group activity classes| 8       |
| Player action classes | 9       |
| Split strategy        | Video-level (no frame leakage)|

### Group Activity Class Distribution

| Class        | Instances |
|--------------|-----------|
| Left Pass    | 826       |
| Right Pass   | 801       |
| Left Spike   | 642       |
| Right Spike  | 623       |
| Left Set     | 633       |
| Right Set    | 644       |
| Left Winpoint| 367       |
| Right Winpoint| 295      |

### Player Action Class Distribution

| Action    | Instances |
|-----------|-----------|
| Standing  | 38,696    |
| Moving    | 5,121     |
| Waiting   | 3,601     |
| Blocking  | 2,458     |
| Digging   | 2,333     |
| Setting   | 1,332     |
| Falling   | 1,241     |
| Spiking   | 1,216     |
| Jumping   | 341       |

Standing accounts for nearly 69% of all player-level labels. The extreme imbalance at the action level (Standing vs. Jumping: ~113:1) makes person-level supervision the harder of the two training stages.

### Dataset Split

- **Train videos**: 1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54
- **Validation videos**: 0 2 8 12 17 19 24 26 27 28 30 33 46 49 51
- **Test videos**: 4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47

The split is performed at the video level to prevent any temporal leakage between train and test sets.

---

## Results

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

---

## Project Structure

```
Volleyball-Activity-Recoginzer/
|
|-- src/                    # Core source code
|   |-- models/             # Model definitions (B8 Person, B8 Group)
|   |-- datasets/           # Data loaders and preprocessing
|   |-- train.py            # Training loop, accepts --config and other CLI flags
|   |-- evaluate.py         # Evaluation and metrics computation
|   |-- utils.py            # Shared helper functions
|
|-- config/                 # YAML configuration files
|   |-- b7_person.yaml      # Person-level model config
|   |-- b7_group.yaml       # Group-level model config
|
|-- Notebooks/              # EDA and analysis notebooks
|-- checkpoints/            # Saved model weights
|-- logs/                   # Training logs (MLflow runs)
|-- data/videos_dataset      # Volleyball Dataset (downloaded separately)
|-- results/                 # Output plots and evaluation artifacts
|-- docs/                    # Additional documentation
|-- scripts/                 # Training, evaluation, and dataset download scripts
|-- Requirements.txt         # Python dependencies
```

---

## Setup and Installation

**Requirements**: Python 3.8+, CUDA-capable GPU recommended.

```bash
git clone https://github.com/Youssef-Bahaa/Volleyball-Activity-Recoginzer.git
cd Volleyball-Activity-Recoginzer
pip install -r Requirements.txt
```

Download the Volleyball dataset through running the file:

```
 scripts/download_dataset.py
```

This will place the dataset under:

```
Volleyball-Activity-Recoginzer/
    data/
        videos_dataset/
            0/
            1/
            ...
            54/
```

---

## Training

Training proceeds in two sequential stages. The person-level model must be trained first, as its weights are used to initialize the group-level model's feature extractor.

**Stage 1 — Person Model**

```bash
python src/train.py --config config/b7_person.yaml
```

**Stage 2 — Group Model**

```bash
python src/train.py --config config/b7_group.yaml
```

### Command-Line Arguments

`train.py` accepts the following arguments in addition to `--config`:

```bash
python src/train.py \
    --config config/b7_person.yaml \
    --early_stopping \
    --patience 5 \
    --resume checkpoints/b7_person_last.pt \
    --epochs 30 \
    --lr 1e-4
```

| Argument           | Description                                              |
|---------------------|------------------------------------------------------------|
| `--config`          | Path to the YAML configuration file (required)            |
| `--early_stopping`  | Enables early stopping based on validation loss            |
| `--patience`        | Number of epochs to wait for improvement before stopping   |
| `--resume`           | Path to a checkpoint to resume training from               |
| `--epochs`           | Override the number of training epochs set in the config   |
| `--lr`               | Override the learning rate set in the config                |

Experiment tracking is handled via MLflow. Logs and metrics are saved under `logs/` and can be visualized with:

```bash
mlflow ui
```

---

## Configuration

All hyperparameters are controlled through YAML files in `config/`. Key parameters:

| Parameter         | Description                              |
|-------------------|------------------------------------------|
| `backbone`        | CNN backbone (`resnet50`)               |
| `lstm_hidden_size`| Hidden dimension of the LSTM            |
| `num_lstm_layers` | Number of LSTM layers                   |
| `crop_size`       | Player crop resolution (224x224)        |
| `T`               | Number of frames per clip               |
| `lr`              | Initial learning rate                   |
| `weight_decay`    | L2 regularization strength              |
| `label_smoothing` | Label smoothing factor                  |
| `scheduler`       | LR scheduler type (`ReduceLROnPlateau`) |

---

## Citation

If you use this implementation or the Volleyball Dataset, please cite the original paper:

```bibtex
@inproceedings{msibrahiCVPR16deepactivity,
  author    = {Mostafa S. Ibrahim and Srikanth Muralidharan and Zhiwei Deng and Arash Vahdat and Greg Mori},
  title     = {A Hierarchical Deep Temporal Model for Group Activity Recognition.},
  booktitle = {2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}
```

*Implemented by [Youssef Bahaa](https://github.com/Youssef-Bahaa) — Cairo University, Faculty of Computers and Artificial Intelligence.*