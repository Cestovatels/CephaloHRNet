"""
Visualization utilities for cephalometric landmark detection.
Each function saves one self-contained figure.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image, ImageDraw

from data.dataset import LANDMARK_SYMBOLS, LANDMARK_TITLES

sns.set_style('whitegrid')
_C = sns.color_palette('tab10')


# ──────────────────────────────────────────────────────────────────────────────
# Training-curve helpers (one figure per metric)
# ──────────────────────────────────────────────────────────────────────────────

def _epoch_axis(ax, values, total_epochs=None):
    epochs = list(range(1, len(values) + 1))
    if total_epochs:
        ax.set_xlim(1, total_epochs)
    ax.set_xlabel('Epoch')
    return epochs


def plot_loss_curve(train_loss: list, val_loss: list, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = _epoch_axis(ax, train_loss)
    ax.plot(epochs, train_loss, color=_C[0], linewidth=1.5, label='Train')
    ax.plot(epochs, val_loss,   color=_C[1], linewidth=1.5, label='Val')
    best = int(np.argmin(val_loss)) + 1
    ax.axvline(best, color='red', linestyle='--', alpha=0.6, label=f'Best epoch {best}')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curve', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_mre_px_curve(mre_px: list, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = _epoch_axis(ax, mre_px)
    ax.plot(epochs, mre_px, color=_C[2], linewidth=1.5)
    best = int(np.argmin(mre_px)) + 1
    ax.axvline(best, color='red', linestyle='--', alpha=0.6, label=f'Best epoch {best}')
    ax.set_ylabel('MRE (pixels)')
    ax.set_title('Validation MRE (pixels)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_mre_mm_curve(mre_mm: list, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = _epoch_axis(ax, mre_mm)
    ax.plot(epochs, mre_mm, color=_C[3], linewidth=1.5)
    best = int(np.argmin(mre_mm)) + 1
    ax.axvline(best, color='red', linestyle='--', alpha=0.6, label=f'Best epoch {best}')
    ax.set_ylabel('MRE (mm)')
    ax.set_title('Validation MRE (mm)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_lr_curve(lr: list, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = _epoch_axis(ax, lr)
    ax.plot(epochs, lr, color=_C[4], linewidth=1.5)
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.set_title('Learning Rate Schedule', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sdr_curve(history: dict, save_path: str):
    """SDR at 2/2.5/3/4 mm thresholds vs epoch."""
    keys    = ['sdr_2.0mm', 'sdr_2.5mm', 'sdr_3.0mm', 'sdr_4.0mm']
    labels  = ['2.0 mm', '2.5 mm', '3.0 mm', '4.0 mm']
    colors  = [_C[0], _C[1], _C[2], _C[3]]

    fig, ax = plt.subplots(figsize=(8, 4))
    for k, lbl, col in zip(keys, labels, colors):
        if k not in history or not history[k]:
            continue
        vals = history[k]
        epochs = list(range(1, len(vals) + 1))
        ax.plot(epochs, vals, linewidth=1.5, color=col, label=lbl)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('SDR (%)')
    ax.set_title('Validation SDR over Training', fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_training_curves(history: dict, save_dir: str):
    """Convenience wrapper – saves one figure per metric."""
    os.makedirs(save_dir, exist_ok=True)
    plot_loss_curve(history['train_loss'], history['val_loss'],
                    os.path.join(save_dir, 'loss_curve.png'))
    plot_mre_px_curve(history['val_mre_px'],
                      os.path.join(save_dir, 'mre_px_curve.png'))
    if history.get('val_mre_mm') and not all(np.isnan(v) for v in history['val_mre_mm']):
        mre_mm_clean = [v for v in history['val_mre_mm'] if not np.isnan(v)]
        plot_mre_mm_curve(history['val_mre_mm'],
                          os.path.join(save_dir, 'mre_mm_curve.png'))
    plot_lr_curve(history['lr'], os.path.join(save_dir, 'lr_curve.png'))
    plot_sdr_curve(history,      os.path.join(save_dir, 'sdr_curve.png'))


# ──────────────────────────────────────────────────────────────────────────────
# Per-landmark analysis
# ──────────────────────────────────────────────────────────────────────────────

_PALETTE = sns.color_palette('husl', 29)


def plot_per_landmark_errors(per_lm_mre_mm: np.ndarray, save_path: str,
                              title='Per-Landmark MRE'):
    """Horizontal bar chart sorted by error."""
    fig, ax = plt.subplots(figsize=(9, 10))
    order  = np.argsort(per_lm_mre_mm)
    labels = [f"{LANDMARK_SYMBOLS[i]}  ({LANDMARK_TITLES[LANDMARK_SYMBOLS[i]]})"
              for i in order]
    values = per_lm_mre_mm[order]
    colors = [_PALETTE[i % 29] for i in order]

    bars = ax.barh(range(len(labels)), values, color=colors,
                   edgecolor='white', linewidth=0.4)
    for thr, ls in [(2.0, '-'), (2.5, '--'), (3.0, ':')]:
        ax.axvline(thr, color='red', linestyle=ls, alpha=0.6,
                   linewidth=1.2, label=f'{thr} mm')
    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=7.5)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Mean Radial Error (mm)')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, max(values) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sdr_bars(sdr_dict: dict, save_path: str):
    """SDR grouped bar chart."""
    labels = [k.replace('sdr_', 'SDR @ ') for k in sdr_dict]
    values = list(sdr_dict.values())
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values,
                  color=[_C[0], _C[1], _C[2], _C[3]][:len(values)],
                  edgecolor='white', width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_ylabel('SDR (%)')
    ax.set_title('Successful Detection Rate', fontweight='bold')
    ax.axhline(100, color='gray', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_distribution(errors_mm: np.ndarray, save_path: str):
    """Histogram + KDE of all per-landmark errors."""
    from scipy.stats import gaussian_kde
    flat = errors_mm.flatten()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(flat, bins=60, density=True, alpha=0.55, color=_C[0], edgecolor='white')
    xs = np.linspace(0, np.percentile(flat, 99), 300)
    ax.plot(xs, gaussian_kde(flat)(xs), color=_C[1], linewidth=2)
    for thr, col in [(2.0, 'green'), (2.5, 'orange'), (3.0, 'red'), (4.0, 'purple')]:
        ax.axvline(thr, color=col, linestyle='--', alpha=0.75, label=f'{thr} mm')
    ax.set_xlabel('Radial Error (mm)')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution (all landmarks)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Precision / Recall / F1 / PR curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_f1_curve(errors_mm: np.ndarray, save_path: str,
                  thresholds=None):
    """
    F1 score vs distance threshold.
    For single-point-per-class detection: P = R = SDR → F1 = SDR.
    Shows how detection rate changes with threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 10.0, 200)

    sdrs = [(errors_mm < t).mean() * 100 for t in thresholds]
    best_idx = int(np.argmax(sdrs))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, sdrs, color=_C[0], linewidth=2)
    ax.axvline(thresholds[best_idx], color='red', linestyle='--', alpha=0.6,
               label=f'Best T={thresholds[best_idx]:.2f}mm  F1={sdrs[best_idx]:.1f}%')
    for thr in [2.0, 2.5, 3.0, 4.0]:
        val = (errors_mm < thr).mean() * 100
        ax.axvline(thr, color='gray', linestyle=':', alpha=0.5)
        ax.text(thr + 0.05, val + 1, f'{val:.1f}%', fontsize=8, color='gray')
    ax.set_xlabel('Distance Threshold (mm)')
    ax.set_ylabel('F1 / Precision / Recall = SDR (%)')
    ax.set_title('F1 Score vs. Detection Threshold', fontweight='bold')
    ax.legend()
    ax.set_xlim(0, thresholds[-1])
    ax.set_ylim(0, 102)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pr_curve(errors_mm: np.ndarray, confidences: np.ndarray,
                  save_path: str):
    """
    Confidence-based Precision-Recall curve.

    Parameters
    ----------
    errors_mm   : (B, N) float  – per-landmark errors in mm
    confidences : (B, N) float  – heatmap peak values (higher = more confident)
    """
    CORRECT_THR = 2.0   # mm threshold to call a detection correct

    flat_err  = errors_mm.flatten()
    flat_conf = confidences.flatten()

    sort_idx = np.argsort(-flat_conf)   # descending by confidence
    flat_err  = flat_err[sort_idx]
    flat_conf = flat_conf[sort_idx]

    correct = flat_err < CORRECT_THR
    n_total = len(correct)
    n_pos   = correct.sum()

    precisions, recalls = [1.0], [0.0]
    tp = 0
    for i, c in enumerate(correct):
        if c:
            tp += 1
        precisions.append(tp / (i + 1))
        recalls.append(tp / n_pos if n_pos > 0 else 0.0)

    # np.trapz removed in NumPy 2.0, use np.trapezoid (added in 2.0) or fallback
    _trapz = getattr(np, 'trapezoid', np.trapezoid)
    auc = _trapz(precisions, recalls)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recalls, precisions, color=_C[0], linewidth=2)
    ax.fill_between(recalls, precisions, alpha=0.15, color=_C[0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve  (threshold={CORRECT_THR} mm,  AUC={auc:.3f})',
                 fontweight='bold')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.text(0.05, 0.05, f'AUC = {auc:.4f}', transform=ax.transAxes,
            fontsize=12, fontweight='bold', color=_C[0])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_detection_matrix(errors_mm: np.ndarray, save_path: str,
                           thresholds=(2.0, 2.5, 3.0, 4.0)):
    """
    Heatmap: 29 landmarks × 4 thresholds showing detection rate (%).
    Replaces classical confusion matrix — shows per-landmark difficulty at
    each clinically relevant threshold.
    """
    N = errors_mm.shape[1]
    matrix = np.zeros((N, len(thresholds)), dtype=np.float32)
    for j, t in enumerate(thresholds):
        matrix[:, j] = (errors_mm < t).mean(axis=0) * 100   # (N,)

    labels_y = [f"{LANDMARK_SYMBOLS[i]}  {LANDMARK_TITLES[LANDMARK_SYMBOLS[i]]}"
                for i in range(N)]
    labels_x = [f'{t} mm' for t in thresholds]

    fig, ax = plt.subplots(figsize=(7, 12))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label='Detection Rate (%)')

    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels(labels_x, fontsize=10)
    ax.set_yticks(range(N))
    ax.set_yticklabels(labels_y, fontsize=8)
    ax.set_title('Per-Landmark Detection Rate at Each Threshold', fontweight='bold')

    for i in range(N):
        for j in range(len(thresholds)):
            ax.text(j, i, f'{matrix[i, j]:.0f}', ha='center', va='center',
                    fontsize=7, color='black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(pred_coords: np.ndarray, gt_coords: np.ndarray,
                           save_path: str):
    """
    29×29 landmark swap matrix.
    Cell (i, j): fraction of samples where predicted landmark i
    is nearest to GT landmark j.  Diagonal = correct; off-diagonal = confusion.

    Parameters
    ----------
    pred_coords : (B, N, 2) in any consistent unit (pixels or mm)
    gt_coords   : (B, N, 2) same unit
    """
    B, N, _ = pred_coords.shape
    matrix = np.zeros((N, N), dtype=np.float32)

    for b in range(B):
        for i in range(N):
            dists = np.sqrt(((pred_coords[b, i] - gt_coords[b]) ** 2).sum(-1))
            j = int(np.argmin(dists))
            matrix[i, j] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = matrix / (row_sums + 1e-8)   # row-normalize → probability

    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Fraction of samples')

    ticks = list(range(N))
    ax.set_xticks(ticks);  ax.set_xticklabels(LANDMARK_SYMBOLS, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(ticks);  ax.set_yticklabels(LANDMARK_SYMBOLS, fontsize=8)
    ax.set_xlabel('Nearest GT Landmark')
    ax.set_ylabel('Predicted Landmark')
    ax.set_title('Landmark Confusion Matrix\n'
                 '(row i, col j = fraction where pred[i] is nearest to GT[j])',
                 fontweight='bold')

    # Diagonal annotation
    diag_mean = matrix.diagonal().mean()
    ax.text(0.98, 0.02, f'Mean diagonal: {diag_mean:.3f}',
            transform=ax.transAxes, ha='right', fontsize=10,
            color='navy', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Prediction overlay
# ──────────────────────────────────────────────────────────────────────────────

POINT_RADIUS = 6


def draw_landmarks_on_image(img_np: np.ndarray,
                             pred_coords: np.ndarray,
                             gt_coords: np.ndarray | None,
                             imgsz: int,
                             errors_mm: np.ndarray | None = None) -> np.ndarray:
    """
    Draw predicted (red) and optionally GT (green) landmarks on image.
    Yellow lines connect pred→GT.  Landmark symbol labels on hover.

    Parameters
    ----------
    img_np      : (H, W, 3) uint8
    pred_coords : (N, 2) in scaled-image pixel space
    gt_coords   : (N, 2) or None
    imgsz       : display size
    errors_mm   : (N,) per-landmark mm error or None
    """
    img_pil = Image.fromarray(img_np).resize((imgsz, imgsz), Image.BILINEAR)
    draw = ImageDraw.Draw(img_pil)

    if gt_coords is not None:
        for pred, gt in zip(pred_coords, gt_coords):
            draw.line([(int(pred[0]), int(pred[1])), (int(gt[0]), int(gt[1]))],
                      fill='yellow', width=1)

    for i, pred in enumerate(pred_coords):
        px, py = int(pred[0]), int(pred[1])
        r = POINT_RADIUS
        draw.ellipse([(px - r, py - r), (px + r, py + r)],
                     fill='red', outline='white', width=1)
        lbl = LANDMARK_SYMBOLS[i]
        if errors_mm is not None:
            lbl += f' {errors_mm[i]:.1f}'
        draw.text((px + r + 2, py - 5), lbl, fill='red')

    if gt_coords is not None:
        for i, gt in enumerate(gt_coords):
            gx, gy = int(gt[0]), int(gt[1])
            r = POINT_RADIUS
            draw.ellipse([(gx - r, gy - r), (gx + r, gy + r)],
                         fill='lime', outline='white', width=1)
            draw.text((gx + r + 2, gy - 5), LANDMARK_SYMBOLS[i], fill='lime')

    return np.array(img_pil)


def plot_predictions(images_np: list,
                     pred_coords_list: list,
                     gt_coords_list: list,
                     errors_mm: np.ndarray,
                     save_path: str,
                     imgsz: int = 512,
                     n_cols: int = 4):
    """
    Grid of prediction visualizations (red=pred, green=GT).
    """
    n = min(len(images_np), 12)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = np.array(axes).flatten()

    for i in range(n):
        gt = gt_coords_list[i] if gt_coords_list else None
        ann = draw_landmarks_on_image(images_np[i], pred_coords_list[i],
                                      gt, imgsz)
        axes[i].imshow(ann)
        title = f'MRE: {errors_mm[i]:.2f} mm' if errors_mm is not None else ''
        axes[i].set_title(title, fontsize=9)
        axes[i].axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    handles = [mpatches.Patch(color='red',  label='Predicted'),
               mpatches.Patch(color='lime', label='Ground Truth')]
    fig.legend(handles=handles, loc='upper right', fontsize=10)
    plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
