"""Train KneeNet with site-grouped cross-validation.

Inputs : cache dir (from cache_data.py), labels_soft.csv (from labeler_combine.py combine)
Outputs: <out>/fold{k}.pt (EMA weights + config), oof.csv, cv_report.json, folds.csv

Validation labels: for each label, studies with ev>=0.8 -> positive, ev<=0.2 -> negative, others ignored.
ev_<label> = y_<label> by default; with --eval_labels it is a FIXED label file (e.g. the baseline oof_all.csv), so
training on different targets (v4/v5 calibrated labels) never changes how epochs/models are selected or scored.
Gold studies (full human labels) are also scored separately ("gold" metrics; small n -> wide CIs).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import DataLoader, Dataset

from common import LABELS, set_seed, attach_eval_labels, check_folds
from model import EMA, KneeNet, SoftBCE
from geom import apply_study


# --------------------------------------------------------------------------- data
class KneeDS(Dataset):
    def __init__(self, df, cache_dir, K, train=False, geom=None):
        self.df, self.dir, self.K, self.train = df.reset_index(drop=True), Path(cache_dir), K, train
        self.geom = geom                                    # {study: (codes[S], side)} from geom_table.py, or None
        self.S = 3 * K
        self.y = self.df[[f"y_{l}" for l in LABELS]].values.astype(np.float32)
        self.w = self.df[[f"w_{l}" for l in LABELS]].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def augment(self, x):                                   # x float [S, D, H, W] in 0..255
        S, D, H, W = x.shape
        ang = (torch.rand(S) * 2 - 1) * math.radians(10)
        sc = 1 + (torch.rand(S) * 2 - 1) * 0.15
        tx, ty = (torch.rand(S) * 2 - 1) * 0.08 * 2, (torch.rand(S) * 2 - 1) * 0.08 * 2
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)   # [S,2,3]
        grid = F.affine_grid(theta, (S, D, H, W), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        g = torch.exp((torch.rand(S, 1, 1, 1) * 2 - 1) * 0.3)                 # gamma
        x = 255 * (x / 255).clamp(0, 1) ** g
        x = x * (1 + (torch.rand(S, 1, 1, 1) * 2 - 1) * 0.2) + (torch.rand(S, 1, 1, 1) * 2 - 1) * 10
        x = x + torch.randn_like(x) * torch.rand(S, 1, 1, 1) * 4
        return x.clamp(0, 255)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        arr = np.load(self.dir / f"{r.StudyInstanceUID}.npy", mmap_mode="r")[:].copy()
        if self.geom is not None and r.StudyInstanceUID in self.geom:
            codes, side = self.geom[r.StudyInstanceUID]
            arr = apply_study(arr, [int(r[f"plane_{j}"]) for j in range(self.S)],
                              [int(r[f"present_{j}"]) for j in range(self.S)], codes, side)
        x = torch.from_numpy(arr).float()
        if self.train:
            x = self.augment(x)
        pres = torch.tensor([r[f"present_{j}"] for j in range(self.S)], dtype=torch.long)
        plane = torch.tensor([r[f"plane_{j}"] for j in range(self.S)], dtype=torch.long)
        fluid = torch.tensor([r[f"fluid_{j}"] for j in range(self.S)], dtype=torch.long)
        fat = torch.tensor([r[f"fat_{j}"] for j in range(self.S)], dtype=torch.long)
        return x, pres, plane, fluid, fat, torch.from_numpy(self.y[i]), torch.from_numpy(self.w[i])


# --------------------------------------------------------------------------- metrics
def hard_eval_labels(y):
    t = np.full_like(y, np.nan, dtype=float)
    t[y >= 0.8] = 1
    t[y <= 0.2] = 0
    return t


def macro_auc(t, p):
    """t: [N,12] with NaN = ignore. Returns (macro, per-label list)."""
    aucs = []
    for j in range(t.shape[1]):
        m = ~np.isnan(t[:, j])
        if m.sum() > 1 and len(np.unique(t[m, j])) == 2:
            aucs.append(roc_auc_score(t[m, j], p[m, j]))
        else:
            aucs.append(np.nan)
    return (float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else float("nan")), aucs


def bootstrap_ci(t, p, n=500, seed=0):
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n):
        idx = rng.integers(0, len(t), len(t))
        res.append(macro_auc(t[idx], p[idx])[1])
    res = np.array(res, dtype=float)
    return np.nanpercentile(res, 2.5, axis=0), np.nanpercentile(res, 97.5, axis=0)


# --------------------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for x, pres, plane, fluid, fat, _, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            lg = model(x.to(device), pres.to(device), plane.to(device), fluid.to(device), fat.to(device))
        out.append(torch.sigmoid(lg.float()).cpu().numpy())
    return np.concatenate(out)


def run_epochs(model, ema, loader, val_loader, val_t, loss_fn, opt, sched, epochs, device, scaler, accum,
               patience, val_gold_t=None, tag=""):
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        opt.zero_grad()
        for it, (x, pres, plane, fluid, fat, y, w) in enumerate(loader):
            x, pres, plane, fluid, fat, y, w = [a.to(device) for a in (x, pres, plane, fluid, fat, y, w)]
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                lg = model(x, pres, plane, fluid, fat)
            loss = loss_fn(lg.float(), y, w) / accum
            scaler.scale(loss).backward()
            if (it + 1) % accum == 0 or it + 1 == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
                sched.step(); ema.update(model)
            tot += loss.item() * accum; n += 1
        # evaluate EMA weights
        live = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())
        p = predict(model, val_loader, device)
        score, _ = macro_auc(val_t, p)
        gscore = macro_auc(val_gold_t, p)[0] if val_gold_t is not None else float("nan")
        model.load_state_dict(live)
        print(f"{tag} ep {ep + 1}/{epochs} loss {tot / max(n, 1):.4f} | val macro-AUC {score:.4f} "
              f"| gold {gscore:.4f} | {time.time() - t0:.0f}s", flush=True)
        s = score if score == score else -1
        if s > best:
            best, best_state, bad = s, {k: v.clone() for k, v in ema.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"{tag} early stop"); break
    if best_state is None:                      # every epoch scored NaN (no usable validation labels): keep last EMA
        print(f"{tag} WARNING: no valid validation score; using final EMA weights")
        best_state = {k: v.clone() for k, v in ema.state_dict().items()}
    return best, best_state


def make_sched(opt, total_steps, warm):
    def f(s):
        if s < warm:
            return (s + 1) / max(1, warm)
        pr = (s - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--labels", required=True, help="labels_soft.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only_folds", type=int, nargs="*", default=None)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--weights_path", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr_backbone", type=float, default=5e-5)
    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--smoothing", type=float, default=0.05)
    ap.add_argument("--train_slices", type=int, default=None)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--gold_ft_epochs", type=int, default=3)
    ap.add_argument("--min_gold_ft", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_labels", default=None,
                    help="CSV with StudyInstanceUID + y_<label> (+ optional fold) used ONLY for validation/report")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--head", default="series", choices=["series", "tokens"],
                    help="series = v5 (one pooled token per series); tokens = per-label attention over every window")
    ap.add_argument("--geom_csv", default=None,
                    help="geom.csv from geom_table.py: canonical orientation + right-knee mirroring applied at load time")
    a = ap.parse_args()
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(Path(a.cache_dir) / "cache_index.csv")
    lab = pd.read_csv(a.labels)
    df = lab.merge(idx, on="StudyInstanceUID")
    df = df[df.ok == 1].reset_index(drop=True)
    GEOM = None
    if a.geom_csv:
        g = pd.read_csv(a.geom_csv, dtype={"StudyInstanceUID": str}).fillna({"side": ""})
        cc = [f"code_{j}" for j in range(3 * a.K)]
        GEOM = {s: ([int(v) for v in c], sd) for s, c, sd in zip(g.StudyInstanceUID, g[cc].values, g.side)}
        print(f"geometry: {len(GEOM)} studies | right knees {(g.side == 'R').mean():.1%} | side unknown {(g.side == '').mean():.1%}"
              f" | slots left untouched {(g[cc] < 0).values.mean():.1%}")
    print(f"head: {a.head} | canonical geometry: {GEOM is not None}")
    df = attach_eval_labels(df, a.eval_labels)
    print(f"studies: {len(df)} | gold: {int(df.is_gold.sum())} | sites: {df.site_proxy.nunique()} | "
          f"eval labels: {a.eval_labels or 'same as training labels'}")
    groups = df.site_proxy.fillna("unknown")
    if groups.nunique() >= a.folds:
        splits = list(GroupKFold(n_splits=a.folds).split(df, groups=groups))
    else:
        print("WARNING: fewer site groups than folds -> falling back to random KFold (not site-grouped!)")
        splits = list(KFold(a.folds, shuffle=True, random_state=a.seed).split(df))
    df["fold"] = -1
    for k, (_, va) in enumerate(splits):
        df.loc[va, "fold"] = k
    check_folds(df)
    df[["StudyInstanceUID", "fold", "site_proxy"]].to_csv(out / "folds.csv", index=False)

    oof = np.full((len(df), len(LABELS)), np.nan, dtype=np.float32)
    Yall = df[[f"ev_{l}" for l in LABELS]].values          # FIXED evaluation labels
    for k, (tr, va) in enumerate(splits):
        if a.only_folds is not None and k not in a.only_folds:
            continue
        print(f"\n===== fold {k} | train {len(tr)} val {len(va)} =====")
        dtr, dva = df.iloc[tr], df.iloc[va]
        ytr = dtr[[f"y_{l}" for l in LABELS]].values
        pos = (ytr > 0.5).sum(0).clip(min=1); neg = ((ytr <= 0.5).sum(0)).clip(min=1)
        pw = torch.tensor(np.clip(np.sqrt(neg / pos), 1, 10), dtype=torch.float32, device=device)
        loss_fn = SoftBCE(pw, a.smoothing).to(device)
        model = KneeNet(a.backbone, a.pretrained, a.weights_path, train_slices=a.train_slices, head=a.head).to(device)
        ema = EMA(model, 0.999)
        bb = [p for n, p in model.named_parameters() if n.startswith("enc.")]
        hd = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
        opt = torch.optim.AdamW([{"params": bb, "lr": a.lr_backbone}, {"params": hd, "lr": a.lr_head}], weight_decay=a.wd)
        tl = DataLoader(KneeDS(dtr, a.cache_dir, a.K, True, geom=GEOM), batch_size=a.bs, shuffle=True, num_workers=a.workers,
                        drop_last=len(dtr) > a.bs, persistent_workers=a.workers > 0)
        vl = DataLoader(KneeDS(dva, a.cache_dir, a.K, False, geom=GEOM), batch_size=a.bs, shuffle=False, num_workers=a.workers)
        steps = max(1, a.epochs * math.ceil(len(tl) / a.accum))
        sched = make_sched(opt, steps, warm=max(1, len(tl) // a.accum))
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
        val_t = hard_eval_labels(dva[[f"ev_{l}" for l in LABELS]].values)
        gm = dva.is_gold.values.astype(bool)
        val_gold_t = dva[[f"ev_{l}" for l in LABELS]].values[gm].astype(float) if gm.sum() > 4 else None
        if val_gold_t is not None:
            # gold metric uses only gold rows: evaluate via a mask trick (NaN out non-gold)
            gt = np.full((len(dva), len(LABELS)), np.nan); gt[gm] = val_gold_t; val_gold_t = gt
        best, state = run_epochs(model, ema, tl, vl, val_t, loss_fn, opt, sched, a.epochs, device, scaler, a.accum,
                                 a.patience, val_gold_t, tag=f"[f{k}]")
        model.load_state_dict(state)

        # optional stage 2: fine-tune on gold rows only at low LR
        gtr = dtr[dtr.is_gold]
        if a.gold_ft_epochs > 0 and len(gtr) >= a.min_gold_ft:
            print(f"[f{k}] gold fine-tune on {len(gtr)} studies")
            gl = DataLoader(KneeDS(gtr, a.cache_dir, a.K, True, geom=GEOM), batch_size=a.bs, shuffle=True, num_workers=a.workers)
            opt2 = torch.optim.AdamW(model.parameters(), lr=a.lr_backbone * 0.2, weight_decay=a.wd)
            sch2 = make_sched(opt2, a.gold_ft_epochs * len(gl), 1)
            ema2 = EMA(model, 0.9)
            b2, st2 = run_epochs(model, ema2, gl, vl, val_t, loss_fn, opt2, sch2, a.gold_ft_epochs, device, scaler,
                                 1, 99, val_gold_t, tag=f"[f{k}-gold]")
            if b2 >= best:
                model.load_state_dict(st2)
        oof[va] = predict(model, vl, device)
        torch.save({"state": model.state_dict(), "backbone": a.backbone, "K": a.K,
                    "train_slices": a.train_slices, "head": a.head, "canon": GEOM is not None}, out / f"fold{k}.pt")
        del model, opt, tl, vl
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------- CV report
    done = ~np.isnan(oof[:, 0])
    rep = {"n_studies_scored": int(done.sum()), "eval_labels": a.eval_labels}
    T = hard_eval_labels(Yall)
    Tg = np.full_like(T, np.nan); gm = df.is_gold.values.astype(bool); Tg[gm] = Yall[gm]
    for name, tt in (("labels", T), ("gold", Tg)):
        t_, p_ = tt[done], oof[done]
        macro, per = macro_auc(t_, p_)
        lo, hi = bootstrap_ci(t_, p_, a.bootstrap) if np.isfinite(macro) else (np.full(12, np.nan),) * 2
        rep[name] = {"macro_auc": macro,
                     "per_label": {l: {"auc": per[i], "ci95": [float(lo[i]), float(hi[i])]} for i, l in enumerate(LABELS)}}
    site_rows = []
    for s, g in df[done].groupby("site_proxy"):
        if len(g) >= 10:
            mm = macro_auc(T[g.index], oof[g.index])[0]
            site_rows.append((s, len(g), mm))
    rep["per_site_macro_auc"] = [{"site": s, "n": n, "macro_auc": m} for s, n, m in site_rows]
    vals = [m for _, _, m in site_rows if m == m]
    rep["worst_site_macro_auc"] = float(min(vals)) if vals else None
    o = df[["StudyInstanceUID", "fold", "site_proxy", "is_gold"]].copy()
    for i, l in enumerate(LABELS):
        o[f"p_{l}"] = oof[:, i]; o[f"y_{l}"] = Yall[:, i]; o[f"ytrain_{l}"] = df[f"y_{l}"].values
    o.to_csv(out / "oof.csv", index=False)
    (out / "cv_report.json").write_text(json.dumps(rep, indent=2, default=float))
    print("\nCV macro-AUC (labels):", rep["labels"]["macro_auc"], "| gold:", rep["gold"]["macro_auc"],
          "| worst site:", rep["worst_site_macro_auc"])


if __name__ == "__main__":
    main()
