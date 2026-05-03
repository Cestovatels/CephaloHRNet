<div align="center">

# CephaloHRNet

**End-to-end cephalometric landmark detection with HRNet**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: CEPHA29](https://img.shields.io/badge/Dataset-CEPHA29-orange)](https://www.kaggle.com/datasets/felixtemko/cepha29)

[![GitHub Stars](https://img.shields.io/github/stars/Cestovatels/CephaloHRNet?style=flat&logo=github&color=yellow)](https://github.com/Cestovatels/CephaloHRNet/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Cestovatels/CephaloHRNet?style=flat&logo=github&color=blue)](https://github.com/Cestovatels/CephaloHRNet/forks)
[![GitHub Issues](https://img.shields.io/github/issues/Cestovatels/CephaloHRNet?style=flat&logo=github&color=red)](https://github.com/Cestovatels/CephaloHRNet/issues)
[![Last Commit](https://img.shields.io/github/last-commit/Cestovatels/CephaloHRNet?style=flat&logo=github&color=brightgreen)](https://github.com/Cestovatels/CephaloHRNet/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/Cestovatels/CephaloHRNet?style=flat&logo=github&color=lightgrey)](https://github.com/Cestovatels/CephaloHRNet)

A clean, production-ready PyTorch implementation of **HRNet-W32/W48** for cephalometric landmark detection.  
Single command to train, evaluate, and predict — works on CEPHA29 or any custom dataset.

</div>

---

## ✨ Highlights

- **HRNet-W32 & W48** — high-resolution parallel branch architecture
- **Adaptive Wing Loss** (AWing) + AMP mixed-precision training
- **Generic dataset loader** — single or multi-annotator (averages coordinates automatically)
- **Full evaluation suite** — MRE, SDR, F1 curve, PR curve, confusion matrix, detection matrix
- **Auto-organized outputs** — `runs/train/`, `runs/valid/`, `runs/test/`, `runs/predict/`
- **YOLO-style CLI** — all hyperparameters via command line, no config editing needed

---

## 📊 Results

Evaluated on **CEPHA29** (150 test images, Senior Orthodontist annotations):

| Model | Epochs | MRE (mm) ↓ | SDR @ 2mm ↑ | SDR @ 2.5mm ↑ | SDR @ 3mm ↑ | SDR @ 4mm ↑ |
|:-----:|:------:|:----------:|:-----------:|:-------------:|:-----------:|:-----------:|
| HRNet-W48 | 73 | **1.37** | 80.83% | 87.20% | 91.15% | 95.08% |

> Training is still in progress (300 epochs total). Results will be updated.

---

## 🛠️ Installation

```bash
git clone https://github.com/Cestovatels/CephaloHRNet.git
cd CephaloHRNet

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

> **GPU:** Install the CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) **before** running pip install.

---

## 📁 Dataset Structure

### CEPHA29 (built-in support)

```
Dataset/
├── cephalogram_machine_mappings.csv     ← pixel size per image (optional)
├── train/
│   ├── Cephalograms/                    ← .png / .jpg / .bmp
│   └── Annotations/
│       └── Cephalometric Landmarks/
│           ├── Junior Orthodontists/    ← *.json
│           └── Senior Orthodontists/   ← *.json  (default)
├── valid/
└── test/
```

Each annotation file (`<image_id>.json`):

```json
{
  "ceph_id": "image_id",
  "landmarks": [
    { "symbol": "A",   "value": { "x": 1412, "y": 1280 } },
    { "symbol": "ANS", "value": { "x": 1440, "y": 1243 } }
  ]
}
```

### Custom Dataset

```
MyDataset/
├── train/
│   ├── images/       ← any supported image format
│   └── labels/       ← *.json, same format, filename = image stem
├── valid/
└── test/
```

```bash
python train.py --data MyDataset --annotators labels --img-dir images
```

---

## 🚀 Usage

### 🏋️ Training

```bash
# Quick start — HRNet-W32
python train.py --data Dataset --model hrnet_w32 --epochs 200 --batch 8 --amp --device cuda

# Best config — HRNet-W48 with pretrained backbone
python train.py \
  --data Dataset \
  --model hrnet_w48 \
  --epochs 300 \
  --batch 8 \
  --lr 5e-4 \
  --loss awing \
  --amp \
  --device cuda \
  --pretrained-path hrnetv2_w48_imagenet_pretrained.pth \
  --name my_exp

# Average Junior + Senior annotations
python train.py --data Dataset \
  --annotators "Junior Orthodontists" "Senior Orthodontists" \
  --model hrnet_w48 --epochs 300 --amp

# Resume training
python train.py --data Dataset --resume runs/train/my_exp/last.pt
```

After training, `best.pt` is automatically evaluated on the validation split.

| Checkpoint | Size (W48) | Use |
|-----------|-----------|-----|
| `best.pt` | ~254 MB | inference / evaluation |
| `last.pt` | ~762 MB | resume training |

### 🧪 Evaluation

```bash
# Test split
python test.py --weights runs/train/my_exp/best.pt --data Dataset

# Validation split
python test.py --weights runs/train/my_exp/best.pt --data Dataset --split valid
```

Results are saved to `runs/test/<name>_exp/` or `runs/valid/<name>_exp/`:

| Output file | Description |
|-------------|-------------|
| `results.json` | MRE & SDR summary |
| `per_landmark_mre.png` | Per-landmark error bar chart |
| `sdr.png` | SDR at 2 / 2.5 / 3 / 4 mm thresholds |
| `error_dist.png` | Error histogram + KDE |
| `f1_curve.png` | F1 score vs distance threshold |
| `pr_curve.png` | Precision–Recall curve |
| `detection_matrix.png` | Per-landmark × threshold heatmap |
| `confusion_matrix.png` | Landmark swap matrix |
| `predictions.png` | Sample overlay images |

### 🔍 Prediction

```bash
# Predictions only
python predict.py --weights runs/train/my_exp/best.pt --source path/to/images

# With Ground Truth overlay
python predict.py \
  --weights runs/train/my_exp/best.pt \
  --source Dataset/test/Cephalograms \
  --gt-dir "Dataset/test/Annotations/Cephalometric Landmarks/Senior Orthodontists"

# Export coordinates to CSV
python predict.py --weights runs/train/my_exp/best.pt \
  --source Dataset/test/Cephalograms --save-csv
```

Results saved to `runs/predict/<name>_exp/`.

**Overlay legend:**

| Visual | Meaning |
|--------|---------|
| Colored filled circle | Predicted landmark |
| Green ring | Ground truth landmark |
| Yellow line | Error between prediction and GT |

---

## 📂 Output Structure

```
runs/
├── train/
│   └── my_exp/              ← best.pt, last.pt, training curves
├── valid/
│   └── my_exp_exp/          ← evaluation results & plots
├── test/
│   └── my_exp_exp/
└── predict/
    └── my_exp_exp/          ← annotated images
```

---

## ⚙️ Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `Dataset` | Dataset root directory |
| `--model` | `hrnet_w32` | `hrnet_w32` or `hrnet_w48` |
| `--epochs` | `200` | Training epochs |
| `--batch` | `8` | Batch size |
| `--lr` | `1e-3` | Initial learning rate |
| `--loss` | `awing` | `awing` or `mse` |
| `--amp` | flag | FP16 automatic mixed precision |
| `--annotators` | `Senior Orthodontists` | Annotation directories (multiple = averaged) |
| `--img-dir` | `Cephalograms` | Image subdirectory name |
| `--pretrained-path` | — | Local `.pth` pretrained backbone |
| `--resume` | — | Resume from `last.pt` |
| `--patience` | `50` | Early stopping patience (0 = off) |

---

## 🗂️ Project Structure

```
CephaloHRNet/
├── train.py              ← training script
├── test.py               ← evaluation script
├── predict.py            ← inference script
├── data/
│   ├── dataset.py        ← generic dataset loader
│   └── transforms.py     ← augmentation pipeline
├── models/
│   └── hrnet.py          ← HRNet-W32 / W48
├── utils/
│   ├── losses.py         ← AdaptiveWingLoss, MSEHeatmapLoss
│   ├── metrics.py        ← MRE, SDR, coordinate helpers
│   ├── visualize.py      ← all plot functions
│   └── logger.py         ← CSV + console logger
├── configs/
│   └── default.yaml      ← default hyperparameters
└── requirements.txt
```

---

## 🙏 Acknowledgements

- [HRNet](https://github.com/HRNet/HRNet-Facial-Landmark-Detection) — original high-resolution network architecture
- [CEPHA29](https://github.com/manwaarkhd/CEPHA29) — cephalometric landmark dataset
- [aariz](https://github.com/manwaarkhd/aariz) — cephalometric landmark dataset

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
