"""
HRNet Cephalometric Landmark Detection – Inference Script

Output is always saved to:
    runs/predict/<name>_exp      (first run)
    runs/predict/<name>_exp2     (if already exists, auto-increments)

Usage:
    # Single image, predictions only
    python predict.py --weights runs/train/Final-Train/best.pt --source image.png

    # Folder of images with ground truth overlay
    python predict.py --weights runs/train/Final-Train/best.pt \
        --source Dataset/test/Cephalograms \
        --gt-dir "Dataset/test/Annotations/Cephalometric Landmarks/Senior Orthodontists"

    # Custom experiment name
    python predict.py --weights runs/train/Final-Train/best.pt \
        --source Dataset/test/Cephalograms --name my_run
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from PIL import Image, ImageDraw
import cv2

from models.hrnet import build_hrnet
from utils.metrics import decode_heatmaps, heatmap_to_image_coords, image_to_original_coords
from data.dataset import LANDMARK_SYMBOLS, LANDMARK_TITLES, NUM_LANDMARKS, load_annotation

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

# Distinct colors per landmark (hex)
COLORS = [
    '#FF5722', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0',
    '#00BCD4', '#E91E63', '#8BC34A', '#FFC107', '#607D8B',
    '#F44336', '#3F51B5', '#009688', '#FFEB3B', '#795548',
    '#03A9F4', '#8D6E63', '#9E9E9E', '#CDDC39', '#673AB7',
    '#FF6F00', '#1B5E20', '#880E4F', '#0D47A1', '#BF360C',
    '#004D40', '#311B92', '#B71C1C', '#33691E',
]


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def make_exp_dir(name: str) -> Path:
    """
    Create and return runs/predict/<name>_exp.
    If that already exists → runs/predict/<name>_exp2, _exp3, ...
    """
    root = Path('runs') / 'predict'
    candidate = root / f'{name}_exp'
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    i = 2
    while True:
        candidate = root / f'{name}_exp{i}'
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        i += 1


def parse_args():
    p = argparse.ArgumentParser(
        description='Run inference on cephalometric X-ray images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--weights', type=str, required=True, help='model checkpoint path')
    p.add_argument('--source',  type=str, required=True,
                   help='image file or directory of images')
    p.add_argument('--name',    type=str, default='',
                   help='experiment name for output dir (default: inferred from weights path)')
    p.add_argument('--gt-dir',  type=str, default='',
                   help='directory of JSON annotation files for GT overlay')
    p.add_argument('--device',  type=str, default='', help='cuda device or cpu')
    p.add_argument('--radius',  type=int, default=6,  help='landmark circle radius (pixels)')
    p.add_argument('--no-labels', action='store_true', help='do not draw landmark labels')
    p.add_argument('--save-csv',  action='store_true',
                   help='save landmark coordinates to CSV')
    return p.parse_args()


def select_device(s):
    if s == 'cpu':
        return torch.device('cpu')
    if s == '':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(f'cuda:{s}' if s.isdigit() else s)


def preprocess(img_path, imgsz, device):
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size
    img_resized = img.resize((imgsz, imgsz), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return tensor, orig_w, orig_h


def find_gt_annotation(img_path: Path, gt_dir: Path):
    """Look for a JSON annotation whose stem matches the image stem."""
    json_path = gt_dir / (img_path.stem + '.json')
    if json_path.exists():
        return json_path
    # Fallback: search case-insensitively
    for p in gt_dir.glob('*.json'):
        if p.stem.lower() == img_path.stem.lower():
            return p
    return None


def draw_result(img_path, pred_coords, gt_coords, radius, draw_labels):
    """
    Draw predicted (red) and optionally GT (green) landmarks on the original image.
    Yellow lines connect pred to GT when both are present.

    pred_coords : (N, 2) float  in original pixel space (x, y)
    gt_coords   : (N, 2) float  or None
    """
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Draw yellow error lines first (behind dots)
    if gt_coords is not None:
        for i in range(len(pred_coords)):
            px, py = float(pred_coords[i, 0]), float(pred_coords[i, 1])
            gx, gy = float(gt_coords[i, 0]),   float(gt_coords[i, 1])
            draw.line([(px, py), (gx, gy)], fill=(255, 215, 0), width=2)

    for i in range(len(pred_coords)):
        color = hex_to_rgb(COLORS[i % len(COLORS)])
        sym   = LANDMARK_SYMBOLS[i]

        # ── Predicted landmark (filled circle, landmark color) ────────
        px, py = float(pred_coords[i, 0]), float(pred_coords[i, 1])
        r = radius
        draw.ellipse([(px-r, py-r), (px+r, py+r)], fill=color, outline='white', width=1)
        if draw_labels:
            draw.text((px + r + 2, py - 6), sym, fill=color)

        # ── Ground truth landmark (green ring) ────────────────────────
        if gt_coords is not None:
            gx, gy = float(gt_coords[i, 0]), float(gt_coords[i, 1])
            rg = radius + 2
            draw.ellipse([(gx-rg, gy-rg), (gx+rg, gy+rg)],
                         fill=None, outline=(0, 220, 60), width=3)
            if draw_labels:
                draw.text((gx + rg + 2, gy - 6), sym, fill=(0, 220, 60))

    return img


@torch.no_grad()
def predict_single(model, img_path, imgsz, device):
    tensor, orig_w, orig_h = preprocess(img_path, imgsz, device)

    with autocast('cuda', enabled=device.type == 'cuda'):
        hm_pred = model(tensor)

    hm_coords   = decode_heatmaps(hm_pred.float().cpu())           # (1, N, 2)
    img_coords  = heatmap_to_image_coords(hm_coords, imgsz)        # (1, N, 2)
    orig_size   = torch.tensor([[orig_h, orig_w]], dtype=torch.float32)
    orig_coords = image_to_original_coords(img_coords, orig_size, imgsz)  # (1, N, 2)

    return orig_coords[0].numpy(), orig_w, orig_h


def main():
    args   = parse_args()
    device = select_device(args.device)

    # ── Load model ────────────────────────────────────────────────────
    ckpt       = torch.load(args.weights, map_location=device)
    train_args = ckpt.get('args', {})
    model_name = train_args.get('model', 'hrnet_w32')
    imgsz      = train_args.get('imgsz', 512)

    model = build_hrnet(model_name, num_landmarks=NUM_LANDMARKS)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()
    print(f"Loaded {model_name} from {args.weights}")

    # ── Collect images ────────────────────────────────────────────────
    source = Path(args.source)
    if source.is_file():
        img_paths = [source]
    elif source.is_dir():
        img_paths = sorted(p for p in source.iterdir()
                           if p.suffix.lower() in IMG_EXTS)
    else:
        print(f"Error: source not found: {args.source}")
        sys.exit(1)

    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    if gt_dir and not gt_dir.is_dir():
        print(f"Warning: --gt-dir not found: {gt_dir} — GT overlay disabled")
        gt_dir = None

    # ── Output directory ──────────────────────────────────────────────
    name    = args.name if args.name else Path(args.weights).parent.name
    out_dir = make_exp_dir(name)

    print(f"Found {len(img_paths)} image(s)"
          + (f"  |  GT directory: {gt_dir}" if gt_dir else ""))
    print(f"Output dir  : {out_dir}")

    # ── CSV setup ─────────────────────────────────────────────────────
    if args.save_csv:
        import csv
        csv_path = out_dir / 'predictions.csv'
        csv_file = open(csv_path, 'w', newline='')
        header = ['filename'] + [f'{sym}_x' for sym in LANDMARK_SYMBOLS] + \
                                 [f'{sym}_y' for sym in LANDMARK_SYMBOLS]
        writer = csv.writer(csv_file)
        writer.writerow(header)

    # ── Predict & draw ────────────────────────────────────────────────
    gt_missing = 0
    for img_path in img_paths:
        pred_coords, orig_w, orig_h = predict_single(model, img_path, imgsz, device)

        # Load GT if a gt_dir was given
        gt_coords = None
        if gt_dir is not None:
            ann_path = find_gt_annotation(img_path, gt_dir)
            if ann_path is not None:
                gt_coords = load_annotation(ann_path)   # (N, 2) x,y in original pixels
            else:
                gt_missing += 1

        ann_img = draw_result(img_path, pred_coords, gt_coords,
                              args.radius, not args.no_labels)
        out_path = out_dir / img_path.name
        ann_img.save(str(out_path))

        status = "pred+GT" if gt_coords is not None else "pred only"
        print(f"  [{status}] {img_path.name} -> {out_path}")

        if args.save_csv:
            row = [img_path.name] + list(pred_coords[:, 0]) + list(pred_coords[:, 1])
            writer.writerow(row)

    if args.save_csv:
        csv_file.close()
        print(f"\nCoordinates saved -> {csv_path}")

    if gt_dir and gt_missing:
        print(f"\nWarning: GT annotation not found for {gt_missing} image(s)")

    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
