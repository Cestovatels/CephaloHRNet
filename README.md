# CephaloHRNet — Cephalometric Landmark Detection

A clean, end-to-end PyTorch implementation of **HRNet** for cephalometric landmark detection.  
Designed for easy use with any dataset that follows the described structure.

- HRNet-W32 and HRNet-W48 variants
- Adaptive Wing Loss (AWing) + AMP training
- Full evaluation pipeline: MRE, SDR, F1, PR curve, confusion matrix
- YOLO-style CLI — single command to train, test, or predict

---

## Results (CEPHA29 — Senior Orthodontists)

| Model | Epochs | MRE (mm) | SDR @ 2mm | SDR @ 2.5mm | SDR @ 3mm | SDR @ 4mm |
|-------|--------|----------|-----------|-------------|-----------|-----------|
| HRNet-W48 | 73 | 1.37 | 80.83% | 87.20% | 91.15% | 95.08% |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/CephaloHRNet.git
cd CephaloHRNet
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

> **GPU note:** Install the CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

## Dataset Structure

```
Dataset/
├── cephalogram_machine_mappings.csv   # pixel_size per image (optional)
├── train/
│   ├── Cephalograms/                  # .png / .jpg / .bmp images
│   └── Annotations/
│       └── Cephalometric Landmarks/
│           ├── Junior Orthodontists/  # *.json
│           └── Senior Orthodontists/  # *.json  ← default
├── valid/   (same structure)
└── test/    (same structure)
```

Each JSON annotation:
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
│   ├── images/
│   └── labels/      # *.json — same format, file stem = image stem
├── valid/
└── test/
```

```bash
python train.py --data MyDataset --annotators labels --img-dir images
```

---

## Training

```bash
# HRNet-W32, quick start
python train.py --data Dataset --model hrnet_w32 --epochs 200 --batch 8 --amp --device cuda

# HRNet-W48, best config
python train.py \
  --data Dataset \
  --model hrnet_w48 \
  --epochs 300 \
  --batch 8 \
  --imgsz 512 \
  --lr 5e-4 \
  --loss awing \
  --amp \
  --device cuda \
  --pretrained-path hrnetv2_w48_imagenet_pretrained.pth \
  --project runs/train \
  --name my_exp

# Average Junior + Senior annotations
python train.py --data Dataset \
  --annotators "Junior Orthodontists" "Senior Orthodontists" \
  --model hrnet_w48 --epochs 300 --amp

# Resume from checkpoint
python train.py --data Dataset --resume runs/train/my_exp/last.pt
```

Training automatically evaluates `best.pt` on the validation split when finished.  
Checkpoints saved to `runs/train/<name>/`:
- `best.pt` — best validation MRE (model weights only, ~254 MB for W48)
- `last.pt` — last epoch, full checkpoint for resuming (~762 MB for W48)

---

## Evaluation

```bash
# Test split (default)
python test.py --weights runs/train/my_exp/best.pt --data Dataset

# Validation split
python test.py --weights runs/train/my_exp/best.pt --data Dataset --split valid
```

Results saved to `runs/test/<name>_exp/` or `runs/valid/<name>_exp/`:

| File | Description |
|------|-------------|
| `results.json` | MRE, SDR summary |
| `per_landmark_mre.png` | Per-landmark error bar chart |
| `sdr.png` | SDR at 2 / 2.5 / 3 / 4 mm |
| `error_dist.png` | Error histogram + KDE |
| `f1_curve.png` | F1 vs threshold |
| `pr_curve.png` | Precision–Recall curve |
| `detection_matrix.png` | Per-landmark × threshold heatmap |
| `confusion_matrix.png` | Landmark swap matrix |
| `predictions.png` | Sample overlay images |

---

## Prediction

```bash
# Predictions only
python predict.py --weights runs/train/my_exp/best.pt --source Dataset/test/Cephalograms

# With Ground Truth overlay (predicted = colored dot, GT = green ring, error = yellow line)
python predict.py \
  --weights runs/train/my_exp/best.pt \
  --source Dataset/test/Cephalograms \
  --gt-dir "Dataset/test/Annotations/Cephalometric Landmarks/Senior Orthodontists"

# Save coordinates to CSV
python predict.py --weights runs/train/my_exp/best.pt \
  --source Dataset/test/Cephalograms --save-csv
```

Results saved to `runs/predict/<name>_exp/`.

---

## Key Arguments

### `train.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `Dataset` | Dataset root |
| `--model` | `hrnet_w32` | `hrnet_w32` or `hrnet_w48` |
| `--epochs` | `200` | Training epochs |
| `--batch` | `8` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--loss` | `awing` | `awing` or `mse` |
| `--amp` | flag | FP16 mixed precision |
| `--annotators` | `Senior Orthodontists` | Annotation directories (1 = direct use, 2+ = averaged) |
| `--img-dir` | `Cephalograms` | Image subdirectory |
| `--pretrained-path` | — | Local `.pth` pretrained backbone |
| `--resume` | — | Resume from `last.pt` |

### `test.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--weights` | required | Checkpoint path |
| `--split` | `test` | `train` / `valid` / `test` |
| `--name` | auto | Output folder name |

### `predict.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--weights` | required | Checkpoint path |
| `--source` | required | Image or folder |
| `--gt-dir` | — | Annotation folder for GT overlay |
| `--save-csv` | flag | Export coordinates to CSV |

---

## Project Structure

```
CephaloHRNet/
├── train.py              # Training script
├── test.py               # Evaluation script
├── predict.py            # Inference script
├── data/
│   ├── dataset.py        # CephaloDataset — generic loader
│   └── transforms.py     # Augmentation pipeline
├── models/
│   └── hrnet.py          # HRNet-W32 / W48
├── utils/
│   ├── losses.py         # AdaptiveWingLoss, MSEHeatmapLoss
│   ├── metrics.py        # MRE, SDR, decode helpers
│   ├── visualize.py      # All plot functions
│   └── logger.py         # CSV + console logger
├── configs/
│   └── default.yaml      # Default hyperparameters
└── requirements.txt
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
