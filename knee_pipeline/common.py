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


def attach_eval_labels(df, eval_path, labels=LABELS):
    """Add ev_<label> columns = FIXED evaluation labels, independent of the labels used for training.

    eval_path=None  -> ev_<label> = y_<label> (old behaviour).
    eval_path=csv   -> StudyInstanceUID + y_<label> columns (e.g. the baseline oof_all.csv or the old labels_soft.csv);
                       a 'fold' column, if present, is kept as ev_fold so check_folds() can verify the split.
    Why: model selection (best epoch, early stopping) and the CV report must not move when the TRAINING targets
    change (e.g. calibrated v4 labels have 'present' = 0.73, below the 0.8 'positive' threshold)."""
    import pandas as pd
    if not eval_path:
        for l in labels:
            df[f"ev_{l}"] = df[f"y_{l}"]
        return df
    ev = pd.read_csv(eval_path)
    need = [f"y_{l}" for l in labels]
    miss = [x for x in need if x not in ev.columns]
    if miss:
        raise ValueError(f"--eval_labels {eval_path} lacks columns {miss[:3]}...")
    keep = ["StudyInstanceUID"] + need + (["fold"] if "fold" in ev.columns else [])
    ev = ev[keep].drop_duplicates("StudyInstanceUID").rename(columns={f"y_{l}": f"ev_{l}" for l in labels})
    ev = ev.rename(columns={"fold": "ev_fold"})
    out = df.merge(ev, on="StudyInstanceUID", how="left")
    n_missing = int(out[f"ev_{labels[0]}"].isna().sum())
    if n_missing:
        raise ValueError(f"{n_missing} training studies have no row in --eval_labels; scores would not be comparable")
    return out


def check_folds(df):
    """If the eval-label file carried a fold column, the fold split here must be identical (same fold 0)."""
    if "ev_fold" not in df.columns:
        return 0
    bad = int((df["ev_fold"].notna() & (df["fold"] != df["ev_fold"])).sum())
    if bad:
        raise ValueError(f"fold assignment differs from the eval-label file for {bad} studies: results would not be "
                         "comparable with the baseline. Use the same cache index / label rows / --folds.")
    return bad
