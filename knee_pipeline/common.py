"""Shared constants and helpers."""
import glob
import os
import random
from pathlib import Path

import numpy as np

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PLANES = ["Sagittal", "Coronal", "Axial"]
PLANE_ID = {p: i for i, p in enumerate(PLANES)}


def find_data_dir(max_depth=4):
    """Folder that contains train.csv (Kaggle: /kaggle/input/...). Set KNEE_DATA_DIR to override."""
    env = os.environ.get("KNEE_DATA_DIR")
    if env:
        return Path(env)
    roots = [os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"), "."]
    skip = {"train_series", "test_series", ".git", "__pycache__"}
    for root in roots:
        if not os.path.isdir(root):
            continue
        queue = [(root, 0)]
        while queue:
            cur, depth = queue.pop(0)
            try:
                entries = list(os.scandir(cur))
            except OSError:
                continue
            if any(e.name == "train.csv" and e.is_file() for e in entries):
                return Path(cur)
            if depth < max_depth:
                queue += [(e.path, depth + 1) for e in entries if e.is_dir() and e.name not in skip]
    raise FileNotFoundError("train.csv not found. Set KNEE_DATA_DIR.")


def work_dir(name):
    base = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("work")
    p = base / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
