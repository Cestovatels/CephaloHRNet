"""
HRNet Cephalometric Landmark Detection – Training Script

Usage (YOLO-style):
    python train.py --data Dataset --model hrnet_w32 --epochs 200 --batch 8 \
                    --imgsz 512 --lr 1e-3 --optimizer AdamW --device cuda \
                    --project runs/train --name exp1 --amp --pretrained

    # Senior only (default)
    python train.py --data Dataset --model hrnet_w48 --epochs 300 --batch 4 \
                    --imgsz 512 --lr 5e-4 --loss awing --amp

    # Average Junior + Senior
    python train.py --data Dataset --model hrnet_w48 --epochs 300 \
                    --annotators "Junior Orthodontists" "Senior Orthodontists"

    # Custom dataset (single annotator, custom image folder)
    python train.py --data /path/to/custom --annotators annotations --img-dir images
"""

import os
import sys
import argparse
import random
import time
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset import CephaloDataset
from data.transforms import build_transforms
from models.hrnet import build_hrnet
from utils.losses import build_loss
from utils.metrics import (
    decode_heatmaps, heatmap_to_image_coords,
    image_to_original_coords, compute_mre, compute_sdr,
)
from utils.visualize import plot_all_training_curves
from utils.logger import MetricLogger


# ──────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='HRNet Cephalometric Landmark Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument('--data', type=str, default='Dataset',
                   help='path to dataset root (contains train/valid/test folders)')
    p.add_argument('--workers', type=int, default=0,
                   help='number of DataLoader worker processes')

    # Model
    p.add_argument('--model', type=str, default='hrnet_w32',
                   choices=['hrnet_w32', 'hrnet_w48'],
                   help='HRNet width variant')
    p.add_argument('--pretrained', action='store_true',
                   help='load ImageNet pretrained HRNet weights via timm (auto-download)')
    p.add_argument('--pretrained-path', type=str, default='',
                   help='path to local .pth pretrained backbone weights (alternative to --pretrained)')
    p.add_argument('--resume', type=str, default='',
                   help='path to checkpoint to resume from')

    # Training hyperparameters
    p.add_argument('--epochs', type=int, default=200, help='total training epochs')
    p.add_argument('--batch', type=int, default=8, help='batch size')
    p.add_argument('--imgsz', type=int, default=512, help='input image size (square)')
    p.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    p.add_argument('--min-lr', type=float, default=1e-6, help='minimum LR for cosine schedule')
    p.add_argument('--warmup-epochs', type=int, default=5, help='linear warmup epochs')
    p.add_argument('--lr-scheduler', type=str, default='cosine',
                   choices=['cosine', 'step', 'onecycle'],
                   help='learning rate scheduler')
    p.add_argument('--step-size', type=int, default=60, help='StepLR step size in epochs')
    p.add_argument('--step-gamma', type=float, default=0.3, help='StepLR gamma')

    # Optimizer
    p.add_argument('--optimizer', type=str, default='AdamW',
                   choices=['AdamW', 'Adam', 'SGD'],
                   help='optimizer type')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    p.add_argument('--grad-clip', type=float, default=1.0,
                   help='gradient clipping max norm (0 = disabled)')

    # Loss
    p.add_argument('--loss', type=str, default='awing',
                   choices=['awing', 'mse'],
                   help='heatmap loss function')
    p.add_argument('--sigma', type=float, default=3.0,
                   help='Gaussian heatmap sigma (pixels at heatmap scale)')

    # Augmentation
    p.add_argument('--no-augment', action='store_true', help='disable training augmentation')
    p.add_argument('--rotation', type=float, default=15.0, help='max rotation angle (degrees)')
    p.add_argument('--scale-low', type=float, default=0.85, help='random scale lower bound')
    p.add_argument('--scale-high', type=float, default=1.15, help='random scale upper bound')

    # Annotations
    p.add_argument('--annotators', type=str, nargs='+',
                   default=['Senior Orthodontists'],
                   help=(
                       'one or more annotation directory names/paths. '
                       'Single entry: used directly. '
                       'Multiple entries: annotations are averaged. '
                       'Relative paths are resolved under '
                       '<data>/<split>/Annotations/Cephalometric Landmarks/ '
                       'or <data>/<split>/; absolute paths are used as-is. '
                       'Examples: "Senior Orthodontists" | '
                       '"Junior Orthodontists" "Senior Orthodontists" | '
                       '/abs/path/to/ann'
                   ))
    p.add_argument('--img-dir', type=str, default='Cephalograms',
                   help='image subdirectory name or path inside <data>/<split>/')

    # Hardware
    p.add_argument('--device', type=str, default='',
                   help='cuda device id ("0", "0,1", "cpu"); empty = auto-select')
    p.add_argument('--amp', action='store_true',
                   help='use automatic mixed precision (FP16) training')
    p.add_argument('--seed', type=int, default=42, help='random seed')

    # Output
    p.add_argument('--project', type=str, default='runs/train', help='output project directory')
    p.add_argument('--name', type=str, default='exp', help='experiment name')
    p.add_argument('--patience', type=int, default=50,
                   help='early stopping patience in epochs (0 = disabled)')

    return p.parse_args()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_str: str) -> torch.device:
    if device_str == 'cpu':
        return torch.device('cpu')
    if device_str == '':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(f'cuda:{device_str}' if device_str.isdigit() else device_str)


def make_run_dir(project: str, name: str) -> Path:
    base = Path(project) / name
    if base.exists():
        idx = 2
        while True:
            candidate = Path(project) / f'{name}{idx}'
            if not candidate.exists():
                base = candidate
                break
            idx += 1
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_lr(optimizer) -> float:
    return optimizer.param_groups[0]['lr']


def cosine_lr(epoch, total_epochs, warmup_epochs, lr_max, lr_min):
    if epoch < warmup_epochs:
        return lr_max * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device, imgsz):
    model.eval()
    total_loss = 0.0
    all_pred, all_gt, all_ps = [], [], []

    for batch in loader:
        imgs = batch['image'].to(device)
        hm_gt = batch['heatmaps'].to(device, dtype=torch.float32)

        with autocast('cuda', enabled=device.type == 'cuda'):
            hm_pred = model(imgs)

        # Cast to float32 before loss: AMP outputs float16 which can overflow
        # causing inf → nan in AWingLoss (inf - inf = nan)
        loss = criterion(hm_pred.float(), hm_gt)
        total_loss += loss.item() * imgs.size(0)

        hm_coords = decode_heatmaps(hm_pred.float().cpu())    # (B, N, 2)
        img_coords = heatmap_to_image_coords(hm_coords, imgsz)  # (B, N, 2)
        orig_coords = image_to_original_coords(
            img_coords, batch['orig_size'], imgsz
        )  # (B, N, 2)

        all_pred.append(orig_coords.numpy())
        all_gt.append(batch['landmarks_orig'].numpy())
        all_ps.append(batch['pixel_size'].numpy())

    pred_np = np.concatenate(all_pred, axis=0)
    gt_np = np.concatenate(all_gt, axis=0)
    ps_np = np.concatenate(all_ps, axis=0)

    metrics = compute_mre(pred_np, gt_np, ps_np)
    sdr = compute_sdr(metrics['errors_mm'])
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, metrics, sdr


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    run_dir = make_run_dir(args.project, args.name)

    print(f"\n{'='*60}")
    print(f"  HRNet Cephalometric Landmark Training")
    print(f"{'='*60}")
    print(f"  Model     : {args.model}")
    print(f"  Data      : {args.data}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch     : {args.batch}")
    print(f"  Image size: {args.imgsz}×{args.imgsz}")
    print(f"  LR        : {args.lr}")
    print(f"  Optimizer : {args.optimizer}")
    print(f"  Loss      : {args.loss}")
    print(f"  Device    : {device}")
    print(f"  AMP       : {args.amp}")
    print(f"  Annotators: {args.annotators}")
    print(f"  Img dir   : {args.img_dir}")
    print(f"  Save dir  : {run_dir}")
    print(f"{'='*60}\n")

    # ── Datasets ──────────────────────────────
    train_tf = build_transforms(
        is_train=True, imgsz=args.imgsz, augment=not args.no_augment,
        rotation=args.rotation, scale=(args.scale_low, args.scale_high),
    )
    val_tf = build_transforms(is_train=False)

    train_ds = CephaloDataset(
        args.data, 'train', imgsz=args.imgsz, sigma=args.sigma,
        transform=train_tf, annotator_dirs=args.annotators,
        img_dir=args.img_dir,
    )
    val_ds = CephaloDataset(
        args.data, 'valid', imgsz=args.imgsz, sigma=args.sigma,
        transform=val_tf, annotator_dirs=args.annotators,
        img_dir=args.img_dir,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == 'cuda',
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == 'cuda',
    )
    print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    # ── Model ─────────────────────────────────
    model = build_hrnet(args.model, num_landmarks=29, pretrained=args.pretrained)
    if args.pretrained_path:
        state = torch.load(args.pretrained_path, map_location='cpu')
        state = state.get('model', state.get('state_dict', state))
        model_dict = model.state_dict()
        matched = {k: v for k, v in state.items()
                   if k in model_dict and 'head' not in k and model_dict[k].shape == v.shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(matched)}/{len(model_dict)} layers from {args.pretrained_path}")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M\n")

    # ── Loss ──────────────────────────────────
    criterion = build_loss(args.loss).to(device)

    # ── Optimizer ─────────────────────────────
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:  # SGD
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay, nesterov=True)

    # ── Scheduler ─────────────────────────────
    if args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.step_gamma)
    elif args.lr_scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs,
            steps_per_epoch=len(train_loader))
    else:
        scheduler = None  # manual cosine with warmup

    scaler = GradScaler('cuda', enabled=args.amp and device.type == 'cuda')

    # ── Resume ────────────────────────────────
    start_epoch = 1
    best_mre = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_mre = ckpt.get('best_mre', float('inf'))
        print(f"Resumed from epoch {start_epoch - 1}, best MRE={best_mre:.4f} px")

    logger = MetricLogger(str(run_dir))
    history = {
        'train_loss': [], 'val_loss': [],
        'val_mre_px': [], 'val_mre_mm': [], 'lr': [],
        'sdr_2.0mm': [], 'sdr_2.5mm': [], 'sdr_3.0mm': [], 'sdr_4.0mm': [],
    }
    no_improve = 0

    # ── Training loop ─────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):

        # Warmup / cosine LR (manual)
        if args.lr_scheduler == 'cosine':
            lr = cosine_lr(epoch - 1, args.epochs, args.warmup_epochs, args.lr, args.min_lr)
            set_lr(optimizer, lr)
        elif args.lr_scheduler == 'step':
            if epoch > 1:
                scheduler.step()
        current_lr = get_lr(optimizer)

        # ── Train epoch ───────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False,
                    bar_format='{l_bar}{bar:20}{r_bar}')

        for batch in pbar:
            imgs = batch['image'].to(device)
            hm_gt = batch['heatmaps'].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=args.amp and device.type == 'cuda'):
                hm_pred = model(imgs)
                loss = criterion(hm_pred, hm_gt)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if args.lr_scheduler == 'onecycle':
                scheduler.step()

            train_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})

        train_loss /= len(train_loader.dataset)

        # ── Validation ────────────────────────
        val_loss, val_metrics, val_sdr = evaluate(
            model, val_loader, criterion, device, args.imgsz)

        mre_px = val_metrics['mre_px']
        mre_mm = val_metrics.get('mre_mm', float('nan'))

        # ── Logging ───────────────────────────
        epoch_metrics = {
            'train_loss': train_loss, 'val_loss': val_loss,
            'val_mre_px': mre_px, 'val_mre_mm': mre_mm,
            'lr': current_lr,
            **val_sdr,
        }
        logger.update(epoch, epoch_metrics)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mre_px'].append(mre_px)
        history['val_mre_mm'].append(mre_mm)
        history['lr'].append(current_lr)
        for k in ['sdr_2.0mm', 'sdr_2.5mm', 'sdr_3.0mm', 'sdr_4.0mm']:
            history[k].append(val_sdr.get(k, 0.0))

        logger.print_epoch(epoch, args.epochs, {
            'train_loss': train_loss, 'val_loss': val_loss,
            'MRE_px': mre_px, 'MRE_mm': mre_mm,
            **val_sdr,
        })

        # ── Checkpoint ────────────────────────
        is_best = mre_px < best_mre
        if is_best:
            best_mre = mre_px
            no_improve = 0
            # best.pt — model weights only (no optimizer → ~1/3 the size, inference use)
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'best_mre': best_mre, 'args': vars(args),
                'val_mre_mm': mre_mm, **val_sdr,
            }, run_dir / 'best.pt')
            print(f"  [BEST] MRE_px={best_mre:.4f} px  MRE_mm={mre_mm:.4f} mm  -> saved best.pt")
        else:
            no_improve += 1

        # last.pt — full checkpoint with optimizer (needed for --resume)
        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_mre': best_mre, 'args': vars(args),
        }, run_dir / 'last.pt')

        # ── Early stopping ────────────────────
        if args.patience > 0 and no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # ── Final plots ───────────────────────────
    print("\nGenerating training curves...")
    plot_all_training_curves(history, str(run_dir))

    # Save args
    import json
    with open(run_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best Val MRE : {best_mre:.4f} px")
    print(f"  Saved to     : {run_dir}")
    print(f"{'='*60}\n")

    # ── Auto-evaluate on validation split ─────
    print(f"{'='*60}")
    print(f"  Auto-evaluating best.pt on validation split ...")
    print(f"{'='*60}\n")
    import subprocess
    subprocess.run([
        sys.executable, 'test.py',
        '--weights', str(run_dir / 'best.pt'),
        '--data',    args.data,
        '--split',   'valid',
        '--name',    args.name,
        '--device',  str(device),
        '--batch',   str(args.batch),
        '--workers', str(args.workers),
    ], check=False)


if __name__ == '__main__':
    main()
