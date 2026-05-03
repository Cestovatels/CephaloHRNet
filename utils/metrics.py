"""
Evaluation metrics for cephalometric landmark detection.

MRE  – Mean Radial Error (Euclidean distance in pixels or mm)
SDR  – Successful Detection Rate at various mm thresholds
"""

import numpy as np
import torch


def decode_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Decode heatmaps to landmark coordinates using argmax.

    Args:
        heatmaps: (B, N, H, W)

    Returns:
        coords: (B, N, 2) in heatmap (x, y) space
    """
    B, N, H, W = heatmaps.shape
    flat = heatmaps.view(B, N, -1)
    idx = flat.argmax(dim=-1)              # (B, N)
    x = (idx % W).float()
    y = (idx // W).float()
    return torch.stack([x, y], dim=-1)    # (B, N, 2)


def heatmap_to_image_coords(heatmap_coords: torch.Tensor, imgsz: int) -> torch.Tensor:
    """
    Scale heatmap coordinates (at stride 4) to image-space coordinates.

    Args:
        heatmap_coords: (B, N, 2) or (N, 2)
        imgsz: input image size (both dims assumed equal)

    Returns:
        image_coords: same shape, scaled by 4
    """
    return heatmap_coords * 4.0


def image_to_original_coords(
    img_coords: torch.Tensor,
    orig_sizes: torch.Tensor,
    imgsz: int,
) -> torch.Tensor:
    """
    Scale coordinates from resized image space back to original image space.

    Args:
        img_coords: (B, N, 2) in [0, imgsz]
        orig_sizes: (B, 2) = [orig_h, orig_w]
        imgsz: training image size

    Returns:
        orig_coords: (B, N, 2) in original pixel space
    """
    scale_w = orig_sizes[:, 1:2] / imgsz     # (B, 1)
    scale_h = orig_sizes[:, 0:1] / imgsz     # (B, 1)
    scale = torch.stack([scale_w, scale_h], dim=-1)  # (B, 1, 2)
    return img_coords * scale


def compute_radial_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Args:
        pred: (B, N, 2) predicted coordinates
        gt:   (B, N, 2) ground-truth coordinates

    Returns:
        errors: (B, N) Euclidean distances in pixels
    """
    diff = pred - gt
    return np.sqrt((diff ** 2).sum(axis=-1))


def compute_mre(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    pixel_sizes: np.ndarray = None,
) -> dict:
    """
    Compute Mean Radial Error (MRE) per sample and overall.

    Args:
        pred_coords: (B, N, 2) in original image pixels
        gt_coords:   (B, N, 2) in original image pixels
        pixel_sizes: (B,) mm/pixel spacing; if None, only pixel MRE returned

    Returns:
        dict with keys: 'mre_px', 'mre_mm' (optional), 'per_sample_mre'
    """
    errors_px = compute_radial_errors(pred_coords, gt_coords)  # (B, N)
    result = {
        'mre_px': float(errors_px.mean()),
        'per_sample_mre': errors_px.mean(axis=1),   # (B,)
        'per_lm_mre': errors_px.mean(axis=0),        # (N,)
        'errors_px': errors_px,
    }
    if pixel_sizes is not None:
        errors_mm = errors_px * pixel_sizes[:, None]  # (B, N)
        result['mre_mm'] = float(errors_mm.mean())
        result['per_lm_mre_mm'] = errors_mm.mean(axis=0)
        result['errors_mm'] = errors_mm
    return result


def compute_sdr(
    errors_mm: np.ndarray,
    thresholds=(2.0, 2.5, 3.0, 4.0),
) -> dict:
    """
    Compute Successful Detection Rate at mm thresholds.

    Args:
        errors_mm: (B, N) errors in mm
        thresholds: iterable of mm thresholds

    Returns:
        dict mapping threshold → SDR percentage
    """
    return {
        f'sdr_{t}mm': float((errors_mm < t).mean() * 100)
        for t in thresholds
    }


def compute_per_landmark_mre(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    pixel_sizes: np.ndarray,
) -> np.ndarray:
    """
    Returns per-landmark MRE in mm, shape (N,).
    """
    errors_px = compute_radial_errors(pred_coords, gt_coords)  # (B, N)
    errors_mm = errors_px * pixel_sizes[:, None]
    return errors_mm.mean(axis=0)
