"""
HRNet Cephalometric Landmark Detection – Evaluation Script

Usage:
    python test.py --weights runs/train/Final-Train/best.pt --data Dataset
    python test.py --weights runs/train/Final-Train/best.pt --data Dataset --split test
    python test.py --weights runs/train/Final-Train/best.pt --data Dataset --split valid

Output is always saved to:
    runs/<split>/<name>_exp      (first run)
    runs/<split>/<name>_exp2     (if already exists, auto-increments)
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from data.dataset import CephaloDataset, LANDMARK_SYMBOLS, LANDMARK_TITLES, NUM_LANDMARKS
from data.transforms import build_transforms
from models.hrnet import build_hrnet
from utils.metrics import (
    decode_heatmaps, heatmap_to_image_coords,
    image_to_original_coords, compute_mre, compute_sdr,
)
from utils.visualize import (
    plot_per_landmark_errors, plot_sdr_bars,
    plot_error_distribution, plot_predictions,
    plot_f1_curve, plot_pr_curve,
    plot_detection_matrix, plot_confusion_matrix,
)


def make_exp_dir(split: str, name: str) -> Path:
    """
    Create and return runs/<split>/<name>_exp.
    If that already exists → runs/<split>/<name>_exp2, _exp3, ...
    """
    root = Path('runs') / split
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
        description='Evaluate HRNet on cephalometric landmark detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--weights', type=str, required=True, help='model checkpoint (.pt)')
    p.add_argument('--data',    type=str, default='Dataset', help='dataset root path')
    p.add_argument('--split',   type=str, default='test',
                   choices=['train', 'valid', 'test'], help='split to evaluate')
    p.add_argument('--name',    type=str, default='',
                   help='experiment name for output dir (default: inferred from weights path)')
    p.add_argument('--batch',   type=int, default=8)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device',  type=str, default='', help='cuda device or cpu')
    p.add_argument('--n-vis',   type=int, default=12,
                   help='number of sample predictions to visualize')
    return p.parse_args()


def select_device(s):
    if s == 'cpu':
        return torch.device('cpu')
    if s == '':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(f'cuda:{s}' if s.isdigit() else s)


def main():
    args   = parse_args()
    device = select_device(args.device)

    # ── Resolve experiment name ───────────────────────────────────────
    # Default: parent folder of the checkpoint (e.g. "Final-Train" from
    #          runs/train/Final-Train/best.pt)
    name = args.name if args.name else Path(args.weights).parent.name

    # ── Output directory ──────────────────────────────────────────────
    out_dir = make_exp_dir(args.split, name)

    # ── Checkpoint ────────────────────────────────────────────────────
    ckpt       = torch.load(args.weights, map_location=device)
    train_args = ckpt.get('args', {})
    model_name = train_args.get('model', 'hrnet_w32')
    imgsz      = train_args.get('imgsz', 512)
    sigma      = train_args.get('sigma', 3.0)
    img_dir    = train_args.get('img_dir', 'Cephalograms')

    # Resolve annotator_dirs: support new format and old checkpoints
    if 'annotators' in train_args:
        annotator_dirs = train_args['annotators']
    elif train_args.get('average_ann', False):
        annotator_dirs = ['Junior Orthodontists', 'Senior Orthodontists']
    elif train_args.get('use_senior', False):
        annotator_dirs = ['Senior Orthodontists']
    else:
        annotator_dirs = ['Junior Orthodontists']

    # ── Model ─────────────────────────────────────────────────────────
    model = build_hrnet(model_name, num_landmarks=NUM_LANDMARKS)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()
    print(f"\nLoaded {model_name}  epoch={ckpt.get('epoch','?')}  from {args.weights}")

    # ── Dataset ───────────────────────────────────────────────────────
    tf      = build_transforms(is_train=False)
    dataset = CephaloDataset(
        args.data, args.split, imgsz=imgsz, sigma=sigma,
        transform=tf, annotator_dirs=annotator_dirs, img_dir=img_dir,
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
    print(f"Evaluating on {args.split}: {len(dataset)} images")
    print(f"Output dir  : {out_dir}\n")

    # ── Inference ─────────────────────────────────────────────────────
    all_pred, all_gt, all_ps = [], [], []
    all_confs   = []
    sample_imgs, sample_pred_scaled, sample_gt_scaled = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating',
                          bar_format='{l_bar}{bar:20}{r_bar}'):
            imgs = batch['image'].to(device)

            with autocast('cuda', enabled=device.type == 'cuda'):
                hm_pred = model(imgs)
            hm_pred = hm_pred.float()

            B, N, H, W = hm_pred.shape
            confs = hm_pred.view(B, N, -1).max(dim=-1).values.cpu().numpy()
            all_confs.append(confs)

            hm_coords   = decode_heatmaps(hm_pred.cpu())
            img_coords  = heatmap_to_image_coords(hm_coords, imgsz)
            orig_coords = image_to_original_coords(img_coords, batch['orig_size'], imgsz)

            all_pred.append(orig_coords.numpy())
            all_gt.append(batch['landmarks_orig'].numpy())
            all_ps.append(batch['pixel_size'].numpy())

            if len(sample_imgs) < args.n_vis:
                n_collect = min(args.n_vis - len(sample_imgs), imgs.size(0))
                for i in range(n_collect):
                    img_t  = batch['image'][i]
                    mean   = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std    = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_np = ((img_t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
                              * 255).astype(np.uint8)
                    sample_imgs.append(img_np)
                    sample_pred_scaled.append(img_coords[i].numpy())
                    sample_gt_scaled.append(batch['landmarks'][i].numpy())

    pred_np = np.concatenate(all_pred,  axis=0)
    gt_np   = np.concatenate(all_gt,    axis=0)
    ps_np   = np.concatenate(all_ps,    axis=0)
    conf_np = np.concatenate(all_confs, axis=0)

    # ── Metrics ───────────────────────────────────────────────────────
    mre_result = compute_mre(pred_np, gt_np, ps_np)
    sdr_result = compute_sdr(mre_result['errors_mm'])
    per_lm_mm  = mre_result['per_lm_mre_mm']

    sp = args.split
    print(f"\n{'='*55}")
    print(f"  Results  ({sp.upper()} set, {len(dataset)} images)")
    print(f"{'='*55}")
    print(f"  MRE (pixels) : {mre_result['mre_px']:.4f} px")
    print(f"  MRE (mm)     : {mre_result['mre_mm']:.4f} mm")
    print(f"  SDR @ 2.0 mm : {sdr_result['sdr_2.0mm']:.2f}%")
    print(f"  SDR @ 2.5 mm : {sdr_result['sdr_2.5mm']:.2f}%")
    print(f"  SDR @ 3.0 mm : {sdr_result['sdr_3.0mm']:.2f}%")
    print(f"  SDR @ 4.0 mm : {sdr_result['sdr_4.0mm']:.2f}%")
    print(f"{'='*55}")
    print(f"\n  Per-Landmark MRE (mm):")
    print(f"  {'Symbol':<8}  {'Title':<30}  {'MRE (mm)':>9}")
    print(f"  {'-'*52}")
    for i, sym in enumerate(LANDMARK_SYMBOLS):
        print(f"  {sym:<8}  {LANDMARK_TITLES[sym]:<30}  {per_lm_mm[i]:>9.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────
    results_path = out_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'split': sp, 'n_images': len(dataset),
            'weights': str(args.weights),
            'epoch': ckpt.get('epoch', '?'),
            'mre_px': float(mre_result['mre_px']),
            'mre_mm': float(mre_result['mre_mm']),
            **{k: float(v) for k, v in sdr_result.items()},
            'per_landmark': {LANDMARK_SYMBOLS[i]: float(per_lm_mm[i])
                             for i in range(NUM_LANDMARKS)},
        }, f, indent=2)
    print(f"\n  Results JSON -> {results_path}")

    # ── Generate all plots ────────────────────────────────────────────
    print(f"\nGenerating plots...")

    plot_per_landmark_errors(
        per_lm_mm,
        str(out_dir / 'per_landmark_mre.png'),
        title=f'Per-Landmark MRE ({sp.upper()} set)',
    )
    plot_sdr_bars(sdr_result,
                  str(out_dir / 'sdr.png'))
    plot_error_distribution(mre_result['errors_mm'],
                            str(out_dir / 'error_dist.png'))
    plot_f1_curve(mre_result['errors_mm'],
                  str(out_dir / 'f1_curve.png'))
    plot_pr_curve(mre_result['errors_mm'], conf_np,
                  str(out_dir / 'pr_curve.png'))
    plot_detection_matrix(mre_result['errors_mm'],
                          str(out_dir / 'detection_matrix.png'))
    plot_confusion_matrix(pred_np, gt_np,
                          str(out_dir / 'confusion_matrix.png'))

    if sample_imgs:
        sample_err_mm = mre_result['errors_mm'][:len(sample_imgs)].mean(axis=1)
        plot_predictions(
            sample_imgs, sample_pred_scaled, sample_gt_scaled, sample_err_mm,
            str(out_dir / 'predictions.png'),
            imgsz=imgsz,
        )

    print(f"  All outputs saved to: {out_dir}")


if __name__ == '__main__':
    main()
