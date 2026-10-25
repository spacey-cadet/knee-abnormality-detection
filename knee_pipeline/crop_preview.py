"""Preview the adaptive crop on random train series (CPU only, no model, no GPU quota).
For each plane: N random series -> top row = current pipeline (no crop), bottom row = adaptive crop,
red box on the top row = crop window. Prints how much of the frame each crop keeps.
Usage (Kaggle): python crop_preview.py --data_dir /kaggle/input/competitions/rsna-knee-abnormality-detection --n 8 --out /kaggle/working/crop_preview
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_data import read_series, crop_box, to_square
from common import PLANES

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--size", type=int, default=160)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="crop_preview")
ap.add_argument("--stats", type=int, default=0, help="also read this many random series per plane and print the distribution of area kept (no images)")
a = ap.parse_args()
data = Path(a.data_dir); root = data / "train_series"
ts = pd.read_csv(data / "train_series.csv")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(a.seed)

for p in PLANES:
    sub = ts[ts.Anatomical_Plane == p]
    sub = sub.iloc[rng.permutation(len(sub))[:a.n * 2]]          # a few spares in case a series is unreadable
    fig, ax = plt.subplots(2, a.n, figsize=(2.2 * a.n, 4.8)); kept = []; col = 0
    for r in sub.itertuples(index=False):
        if col >= a.n: break
        d = root / r.StudyInstanceUID / r.SeriesInstanceUID
        try: v = read_series(d, 24)
        except Exception: v = None
        if v is None: continue
        y0, y1, x0, x1 = crop_box(v)
        h, w = v.shape[1:]; kept.append((y1 - y0) * (x1 - x0) / (h * w))
        mid = v.shape[0] // 2
        before = to_square(v, a.size, a.size)[mid]
        after = to_square(v[:, y0:y1, x0:x1], a.size, a.size)[mid]
        s = max(h, w); oy, ox = (s - h) // 2, (s - w) // 2; sc = a.size / s
        ax[0, col].imshow(before, cmap="gray", vmin=0, vmax=255)
        ax[0, col].add_patch(Rectangle(((x0 + ox) * sc, (y0 + oy) * sc), (x1 - x0) * sc, (y1 - y0) * sc, fill=False, ec="red", lw=1.2))
        ax[1, col].imshow(after, cmap="gray", vmin=0, vmax=255)
        ax[0, col].set_title(f"kept {kept[-1]:.2f}", fontsize=8)
        col += 1
    for x in ax.ravel(): x.axis("off")
    fig.suptitle(f"{p}: top = current (red = crop window), bottom = adaptive crop", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"{p}.png", dpi=110); plt.close(fig)
    k = np.array(kept)
    print(f"{p:9s} n={len(k)} | area kept: median {np.median(k):.2f}, min {k.min():.2f}, uncropped (1.00): {(k > 0.999).sum()}/{len(k)}")
print("images in", out)

if a.stats:
    print(f"\nArea-kept distribution over {a.stats} random series per plane (1.00 = no crop):")
    for p in PLANES:
        sub = ts[ts.Anatomical_Plane == p]; sub = sub.iloc[np.random.default_rng(7).permutation(len(sub))[:a.stats]]; k = []
        for r in sub.itertuples(index=False):
            try: v = read_series(root / r.StudyInstanceUID / r.SeriesInstanceUID, 24)
            except Exception: v = None
            if v is None: continue
            y0, y1, x0, x1 = crop_box(v); k.append((y1 - y0) * (x1 - x0) / (v.shape[1] * v.shape[2]))
        k = np.array(k)
        print(f"{p:9s} n={len(k)} | pct of series cropped: {(k < 0.999).mean() * 100:.0f}% | area kept p10/p25/p50: "
              f"{np.percentile(k, 10):.2f}/{np.percentile(k, 25):.2f}/{np.percentile(k, 50):.2f}")
