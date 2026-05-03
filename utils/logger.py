"""
Training metric logger – tracks loss and metrics per epoch,
writes CSV and prints formatted summaries.
"""

import os
import csv
import time
from collections import defaultdict


class MetricLogger:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.csv_path = os.path.join(save_dir, 'metrics.csv')
        self.history = defaultdict(list)
        self._csv_initialized = False
        self._start = time.time()

    def update(self, epoch: int, metrics: dict):
        metrics['epoch'] = epoch
        metrics['elapsed_min'] = round((time.time() - self._start) / 60, 2)
        for k, v in metrics.items():
            self.history[k].append(v)
        self._write_csv(metrics)

    def _write_csv(self, row: dict):
        fieldnames = list(row.keys())
        write_header = not self._csv_initialized
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow({k: f'{v:.6f}' if isinstance(v, float) else v for k, v in row.items()})

    def print_epoch(self, epoch: int, total: int, metrics: dict):
        elapsed = (time.time() - self._start) / 60
        eta = elapsed / epoch * (total - epoch) if epoch > 0 else 0
        parts = [f"[{epoch:>4}/{total}]", f"Elapsed: {elapsed:5.1f}m", f"ETA: {eta:5.1f}m"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.4f}")
            else:
                parts.append(f"{k}: {v}")
        print("  ".join(parts))

    def get_history(self) -> dict:
        return dict(self.history)

    def best_epoch(self, metric='val_mre_px', mode='min') -> int:
        vals = self.history.get(metric, [])
        if not vals:
            return 1
        if mode == 'min':
            return int(self.history['epoch'][int(import_argmin(vals))])
        return int(self.history['epoch'][int(import_argmax(vals))])


def import_argmin(lst):
    return min(range(len(lst)), key=lst.__getitem__)


def import_argmax(lst):
    return max(range(len(lst)), key=lst.__getitem__)
