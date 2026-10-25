"""Decode knee MRI studies once into compact uint8 arrays [S, D, H, W].

S = 3 planes x K series per plane (Sagittal, Coronal, Axial slots; missing slot = zeros, present=0).
Per series: sort slices by physical position, pick D slices uniformly, clip to the 1st-99.5th percentile,
scale to 0-255 (this IS the per-series normalization), pad to square, resize to HxW.

Disk math (uint8): studies * S * D * H * W bytes.  5000 * 3 * 24 * 160 * 160 = 9.2 GB  (fits /kaggle/working ~20 GB)
                   5000 * 3 * 32 * 224 * 224 = 24 GB  (does NOT fit)  -> choose D/size accordingly.
"""
import argparse
import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

from common import PLANES, find_data_dir


def rank_series(rows):
    """Order candidate series of one plane: fluid+fatsup > fluid > fatsup > other; then more slices."""
    def score(r):
        return (4 * int(r.Fluid_Sensitive == 1) * int(r.Fat_Suppression == 1) + 2 * int(r.Fluid_Sensitive == 1)
                + int(r.Fat_Suppression == 1), getattr(r, "n_slices", 0))
    return sorted(rows.itertuples(index=False), key=score, reverse=True)


def read_series(series_dir, D):
    """-> uint8 [D, h, w] (native in-plane size, normalized) or None."""
    files = [series_dir / f for f in os.listdir(series_dir)]
    heads = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            heads.append((f, ds))
        except Exception:
            continue
    if not heads:
        return None
    try:
        iop = np.array(heads[0][1].ImageOrientationPatient, float)
        nrm = np.cross(iop[:3], iop[3:])
        key = [float(np.dot(np.array(h.ImagePositionPatient, float), nrm)) for _, h in heads]
    except Exception:
        key = [float(getattr(h, "InstanceNumber", i) or i) for i, (_, h) in enumerate(heads)]
    order = np.argsort(key)
    n = len(order)
    pick = order[np.linspace(0, n - 1, D).round().astype(int)]
    vol = []
    for i in pick:
        f, _ = heads[i]
        ds = pydicom.dcmread(f, force=True)
        a = ds.pixel_array.astype(np.float32)
        a = a * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
        vol.append(a)
    shp = max(set(v.shape for v in vol), key=[v.shape for v in vol].count)
    vol = [v if v.shape == shp else np.zeros(shp, np.float32) for v in vol]
    vol = np.stack(vol)
    lo, hi = np.percentile(vol, [1, 99.5])
    if hi <= lo:
        hi = lo + 1
    return (np.clip((vol - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def to_square(vol, H, W):
    D, h, w = vol.shape
    s = max(h, w)
    pad = np.zeros((D, s, s), np.uint8)
    pad[:, (s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = vol
    if s == H == W:
        return pad
    return np.stack([np.asarray(Image.fromarray(x).resize((W, H), Image.BILINEAR)) for x in pad])


def site_proxy(series_dir):
    try:
        f = next(iter(os.listdir(series_dir)))
        ds = pydicom.dcmread(series_dir / f, stop_before_pixels=True, force=True)
        b0 = getattr(ds, "MagneticFieldStrength", "?")
        return f"{str(getattr(ds, 'Manufacturer', '?'))[:12]}|{str(getattr(ds, 'ManufacturerModelName', '?'))[:18]}|{b0}T"
    except Exception:
        return "unknown"


def process_study(root, study, rows, D=24, H=160, W=160, K=1):
    """root = folder containing <study>/<series>/ ; rows = series rows of this study.
    Returns dict(arr[S,D,H,W] uint8, present[S], plane[S], fluid[S], fat[S], site)."""
    S = len(PLANES) * K
    arr = np.zeros((S, D, H, W), np.uint8)
    present = np.zeros(S, np.int8); plane = np.zeros(S, np.int8)
    fluid = np.zeros(S, np.int8); fat = np.zeros(S, np.int8)
    site = "unknown"
    for pi, p in enumerate(PLANES):
        cand = rank_series(rows[rows.Anatomical_Plane == p])
        for k in range(K):
            slot = pi * K + k
            plane[slot] = pi
            if k >= len(cand):
                continue
            r = cand[k]
            d = Path(root) / study / r.SeriesInstanceUID
            try:
                v = read_series(d, D)
                if v is None:
                    continue
                arr[slot] = to_square(v, H, W)
                present[slot] = 1; fluid[slot] = int(r.Fluid_Sensitive); fat[slot] = int(r.Fat_Suppression)
                if site == "unknown":
                    site = site_proxy(d)
            except Exception:
                continue
    return dict(arr=arr, present=present, plane=plane, fluid=fluid, fat=fat, site=site)


def _work(args):
    root, study, rows, out, D, H, W, K = args
    try:
        r = process_study(root, study, rows, D, H, W, K)
        np.save(Path(out) / f"{study}.npy", r["arr"])
        rec = dict(StudyInstanceUID=study, ok=int(r["present"].sum() > 0), site_proxy=r["site"], err="")
        for i in range(len(r["present"])):
            rec[f"present_{i}"], rec[f"plane_{i}"] = int(r["present"][i]), int(r["plane"][i])
            rec[f"fluid_{i}"], rec[f"fat_{i}"] = int(r["fluid"][i]), int(r["fat"][i])
        return rec
    except Exception as e:
        return dict(StudyInstanceUID=study, ok=0, site_proxy="unknown", err=traceback.format_exc()[-200:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--D", type=int, default=24)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--K", type=int, default=1, help="series per plane")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    data = Path(a.data_dir) if a.data_dir else find_data_dir()
    ts = pd.read_csv(data / f"{a.split}_series.csv")
    root = data / f"{a.split}_series"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ts["n_slices"] = [len(os.listdir(root / s / e)) if (root / s / e).exists() else 0
                      for s, e in zip(ts.StudyInstanceUID, ts.SeriesInstanceUID)]
    studies = ts.StudyInstanceUID.unique().tolist()[:a.limit]
    jobs = [(root, s, ts[ts.StudyInstanceUID == s], out, a.D, a.size, a.size, a.K) for s in studies]
    t0 = time.time(); recs = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=2), 1):
            recs.append(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} studies | {time.time() - t0:.0f}s")
    idx = pd.DataFrame(recs)
    idx.to_csv(out / "cache_index.csv", index=False)
    print("failed/empty studies:", int((idx.ok == 0).sum()), "| saved", out / "cache_index.csv")


if __name__ == "__main__":
    main()
