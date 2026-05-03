import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

LANDMARK_SYMBOLS = [
    'A', 'ANS', 'Ar', 'B', 'Co', 'Gn', 'Go',
    'LIA', 'LIT', 'LMT', 'LPM', 'Li', 'Ls',
    'Me', 'N', "N`", 'Or', 'PNS', 'Pn', 'Po',
    'Pog', "Pog`", 'R', 'S', 'Sn',
    'UIA', 'UIT', 'UMT', 'UPM',
]
NUM_LANDMARKS = 29

LANDMARK_TITLES = {
    'A': 'A-point', 'ANS': 'Anterior Nasal Spine', 'Ar': 'Articulare',
    'B': 'B-point', 'Co': 'Condylion', 'Gn': 'Gnathion', 'Go': 'Gonion',
    'LIA': 'Lower Incisor Apex', 'LIT': 'Lower Incisor Tip',
    'LMT': 'Lower Molar Cusp Tip', 'LPM': 'Lower 2nd PM Cusp Tip',
    'Li': 'Labrale inferius', 'Ls': 'Labrale superius',
    'Me': 'Menton', 'N': 'Nasion', "N`": 'Soft Tissue Nasion',
    'Or': 'Orbitale', 'PNS': 'Posterior Nasal Spine', 'Pn': 'Pronasale',
    'Po': 'Porion', 'Pog': 'Pogonion', "Pog`": 'Soft Tissue Pogonion',
    'R': 'Ramus', 'S': 'Sella', 'Sn': 'Subnasale',
    'UIA': 'Upper Incisor Apex', 'UIT': 'Upper Incisor Tip',
    'UMT': 'Upper Molar Cusp Tip', 'UPM': 'Upper 2nd PM Cusp Tip',
}

IMG_EXTS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']


def load_annotation(ann_path):
    """Load a single JSON annotation → (N, 2) float32 array in (x, y) pixel coords."""
    with open(ann_path, 'r') as f:
        data = json.load(f)
    lm_dict = {lm['symbol']: (lm['value']['x'], lm['value']['y'])
               for lm in data['landmarks']}
    return np.array([lm_dict[sym] for sym in LANDMARK_SYMBOLS], dtype=np.float32)


def generate_heatmaps(landmarks_hm, H, W, sigma):
    """
    landmarks_hm : (N, 2)  heatmap-space (x, y) coordinates
    Returns      : (N, H, W) float32 Gaussian heatmaps
    """
    N = len(landmarks_hm)
    heatmaps = np.zeros((N, H, W), dtype=np.float32)
    size = int(6 * sigma + 1)
    x0 = y0 = size // 2
    g = np.exp(-((np.arange(size) - x0) ** 2) / (2 * sigma ** 2))
    gaussian = np.outer(g, g).astype(np.float32)

    for i, (cx, cy) in enumerate(landmarks_hm):
        cx, cy = int(round(cx)), int(round(cy))
        if cx < 0 or cy < 0 or cx >= W or cy >= H:
            continue
        ul = (int(cx - x0), int(cy - y0))
        br = (int(cx + x0), int(cy + y0))
        g_x = (max(0, -ul[0]),     min(br[0], W - 1) - ul[0])
        g_y = (max(0, -ul[1]),     min(br[1], H - 1) - ul[1])
        img_x = (max(0, ul[0]),    min(br[0], W - 1))
        img_y = (max(0, ul[1]),    min(br[1], H - 1))
        if g_x[0] >= g_x[1] or g_y[0] >= g_y[1]:
            continue
        heatmaps[i, img_y[0]:img_y[1] + 1, img_x[0]:img_x[1] + 1] = \
            gaussian[g_y[0]:g_y[1] + 1, g_x[0]:g_x[1] + 1]
    return heatmaps


class CephaloDataset(Dataset):
    """
    Generic cephalometric dataset loader.

    annotator_dirs : list of annotation directory paths.
        - A single entry  → annotations are loaded directly from that directory.
        - Multiple entries → annotations from all directories are averaged per landmark.
        Each path is resolved in order:
          1. Absolute path (used as-is).
          2. Relative to  <root>/<split>/Annotations/Cephalometric Landmarks/<path>
             (CEPHA29 / similar nested structure).
          3. Relative to  <root>/<split>/<path>.
          4. Relative to current working directory.
        Default: ['Senior Orthodontists']

    img_dir : image directory name or path.
        Resolved relative to <root>/<split>/<img_dir>, or as absolute/cwd path.
        Default: 'Cephalograms'

    pixel_size_csv : optional path to a CSV with columns [cephalogram_id, pixel_size].
        If None, the loader searches for <root>/cephalogram_machine_mappings.csv.
        Falls back to default_pixel_size if the CSV is absent or the ID is missing.

    default_pixel_size : mm per pixel used when no CSV entry is found (default 0.1).
    """

    def __init__(
        self,
        root,
        split,
        imgsz=512,
        sigma=3.0,
        transform=None,
        annotator_dirs=None,
        img_dir='Cephalograms',
        pixel_size_csv=None,
        default_pixel_size=0.1,
    ):
        self.root   = Path(root)
        self.split  = split
        self.imgsz  = imgsz
        self.sigma  = sigma
        self.transform = transform
        self.default_pixel_size = default_pixel_size

        split_dir = self.root / split

        if annotator_dirs is None:
            annotator_dirs = ['Senior Orthodontists']

        self.ann_dirs = [self._resolve_ann_dir(split_dir, d) for d in annotator_dirs]
        self.img_dir  = self._resolve_img_dir(split_dir, img_dir)

        # Early sanity checks with helpful messages
        for d in self.ann_dirs:
            if not d.exists():
                raise FileNotFoundError(
                    f"Annotation directory not found: {d}\n"
                    f"  Tip: use --annotators with the correct path relative to "
                    f"{split_dir / 'Annotations' / 'Cephalometric Landmarks'} "
                    f"or provide an absolute path."
                )
        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.img_dir}\n"
                f"  Tip: use --img-dir with the correct name/path."
            )

        self.pixel_size_map = self._load_pixel_size_map(pixel_size_csv)
        self.samples = self._build_samples()

    # ── Directory resolution ──────────────────────────────────────────────────

    def _resolve_ann_dir(self, split_dir: Path, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        # CEPHA29-style nested path
        cepha = split_dir / 'Annotations' / 'Cephalometric Landmarks' / path_str
        if cepha.exists():
            return cepha
        # Relative to split directory
        rel = split_dir / path_str
        if rel.exists():
            return rel
        # Relative to cwd
        if p.exists():
            return p.resolve()
        # Return CEPHA29-style candidate — FileNotFoundError raised later
        return cepha

    def _resolve_img_dir(self, split_dir: Path, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        rel = split_dir / path_str
        if rel.exists():
            return rel
        if p.exists():
            return p.resolve()
        return rel  # FileNotFoundError raised later

    # ── Pixel-size CSV ────────────────────────────────────────────────────────

    def _load_pixel_size_map(self, pixel_size_csv) -> dict:
        candidates = []
        if pixel_size_csv:
            candidates.append(Path(pixel_size_csv))
        candidates.append(self.root / 'cephalogram_machine_mappings.csv')
        for p in candidates:
            if p.exists():
                df = pd.read_csv(p)
                if {'cephalogram_id', 'pixel_size'}.issubset(df.columns):
                    return dict(zip(df['cephalogram_id'], df['pixel_size']))
        return {}

    # ── Sample list ───────────────────────────────────────────────────────────

    def _build_samples(self) -> list:
        samples = []
        skipped = 0
        for ann_file in sorted(self.ann_dirs[0].glob('*.json')):
            ceph_id   = ann_file.stem
            img_path  = self._find_image(ceph_id)
            if img_path is None:
                continue
            # Every annotator directory must have this annotation file
            ann_paths = [d / ann_file.name for d in self.ann_dirs]
            if not all(p.exists() for p in ann_paths):
                skipped += 1
                continue
            samples.append({
                'ceph_id':    ceph_id,
                'img_path':   img_path,
                'ann_paths':  ann_paths,
                'pixel_size': self.pixel_size_map.get(
                    ceph_id, self.default_pixel_size),
            })
        if skipped:
            print(f"  [Dataset/{self.split}] Skipped {skipped} samples: "
                  f"annotation missing in one or more annotator directories.")
        return samples

    def _find_image(self, ceph_id: str):
        for ext in IMG_EXTS:
            p = self.img_dir / (ceph_id + ext)
            if p.exists():
                return p
        return None

    # ── PyTorch interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s['img_path']).convert('RGB')
        orig_w, orig_h = img.size

        # Load all annotations and average (single annotator → no-op mean)
        ann_list       = [load_annotation(p) for p in s['ann_paths']]
        landmarks_orig = np.mean(ann_list, axis=0).astype(np.float32)  # (N, 2)

        img_np  = np.array(img, dtype=np.uint8)
        scale_x = self.imgsz / orig_w
        scale_y = self.imgsz / orig_h
        landmarks_scaled = landmarks_orig.copy()
        landmarks_scaled[:, 0] *= scale_x
        landmarks_scaled[:, 1] *= scale_y

        if self.transform is not None:
            img_np, landmarks_scaled = self.transform(
                img_np, landmarks_scaled, self.imgsz)

        hm_size      = self.imgsz // 4
        landmarks_hm = landmarks_scaled / 4.0
        heatmaps     = generate_heatmaps(landmarks_hm, hm_size, hm_size, self.sigma)

        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return {
            'image':          img_tensor,
            'heatmaps':       torch.from_numpy(heatmaps),
            'landmarks':      torch.from_numpy(landmarks_scaled),
            'landmarks_orig': torch.from_numpy(landmarks_orig),
            'orig_size':      torch.tensor([orig_h, orig_w], dtype=torch.float32),
            'pixel_size':     torch.tensor(s['pixel_size'], dtype=torch.float32),
            'ceph_id':        s['ceph_id'],
        }
