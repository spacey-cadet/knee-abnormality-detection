"""Build geom.csv (canonical orientation code per slot + laterality per study) and print an AUDIT of the cache.

CPU only, headers only (one DICOM header per selected series): minutes, not hours.
  python geom_table.py --split train --out geom.csv [--data_dir ...] [--K 1] [--workers 16] [--limit N]

The audit tells you whether the geometry fix can matter at all:
  * share of series already in the canonical frame (code 0) vs needing transpose / flips / slice reversal,
  * share of series left untouched (-1: oblique, plane disagrees with the CSV, unreadable header),
  * laterality: how many studies resolve from the DICOM tag vs from geometry, how often they disagree,
    share of right knees.
If ~100% of series are code 0 and tag/geometry always agree and nearly all knees are left, the fix is pointless.
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common import PLANES, find_data_dir
from geom import study_geometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    data = Path(a.data_dir) if a.data_dir else find_data_dir()
    root = data / f"{a.split}_series"
    ts = pd.read_csv(data / f"{a.split}_series.csv")
    ts["n_slices"] = [len(os.listdir(root / s / e)) if (root / s / e).exists() else 0     # same tie-break as cache_data.py
                      for s, e in zip(ts.StudyInstanceUID, ts.SeriesInstanceUID)]
    studies = ts.StudyInstanceUID.unique().tolist()[:a.limit]
    groups = {s: g for s, g in ts.groupby("StudyInstanceUID")}

    def one(s):
        try:
            codes, side, src, dis = study_geometry(root, s, groups[s], a.K)
        except Exception:
            codes, side, src, dis = [-1] * (3 * a.K), "", "error", 0
        return dict(StudyInstanceUID=s, side=side, side_src=src, disagree=dis, **{f"code_{j}": c for j, c in enumerate(codes)})

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, studies))
    g = pd.DataFrame(rows)
    g.to_csv(a.out, index=False)

    S = 3 * a.K
    print(f"\n===== GEOMETRY AUDIT ({len(g)} studies, {a.split}) -> {a.out}")
    for pi, p in enumerate(PLANES):
        c = np.concatenate([g[f"code_{pi * a.K + k}"].values for k in range(a.K)])
        n = len(c)
        print(f"{p:9s} untouched(-1) {np.mean(c < 0):6.1%} | already canonical {np.mean(c == 0):6.1%} | "
              f"transpose {np.mean((c >= 0) & (c & 1 > 0)):6.1%} | flip cols {np.mean((c >= 0) & (c & 2 > 0)):6.1%} | "
              f"flip rows {np.mean((c >= 0) & (c & 4 > 0)):6.1%} | reverse slices {np.mean((c >= 0) & (c & 8 > 0)):6.1%}")
    print("\nlaterality source:", g.side_src.replace("", "unknown").value_counts().to_dict())
    print(f"side: L {np.mean(g.side == 'L'):.1%} | R {np.mean(g.side == 'R'):.1%} | unknown {np.mean(g.side == ''):.1%}")
    tagged = g.side_src == "tag"
    print(f"tag vs geometry disagree on {int(g.disagree.sum())} of {int(tagged.sum())} tagged studies"
          f" ({g.disagree.sum() / max(int(tagged.sum()), 1):.1%})")
    cs = [f"code_{j}" for j in range(S)]
    changed = ((g[cs] > 0).any(axis=1) | (g.side == "R")).mean()
    print(f"studies where the fix changes at least one array: {changed:.1%}")


if __name__ == "__main__":
    main()
