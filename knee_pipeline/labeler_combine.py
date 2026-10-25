"""Combine Labeler A (rules) and Labeler B (LLM) into soft training labels, and audit them.

Subcommands
  combine : train.csv + A.csv + B.csv -> labels_soft.csv   (y_<label>, w_<label>, is_gold, lang)
  audit   : write an audit sheet (sample of reports per language) for manual review
  score   : score the filled audit sheet (precision / recall / F1 for A, B, combined) per label and language

Weights (used by the training loss):
  gold rows                 y = gold label, w = 1.0
  other rows                y = confidence-weighted mean of A and B probabilities
                            w = base_w[label] * (0.5 + 0.5 * agreement)
  base_w[label]             from reliability vs gold AUC when enough gold exists, else DEFAULT_BASE_W
"""
import argparse
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import LABELS

DEFAULT_BASE_W = 0.5
# Languages/scripts covered by Labeler A's lexicon. Reports in any other language (e.g. Greek "el",
# Turkish "tr", Polish "pl"...) get Labeler A confidence 0, so Labeler B alone decides.
A_LANGS = {"en", "es", "fr", "de", "pt", "it", "ca", "ru", "zh-cn", "zh-tw", "latin", "cyrillic", "cjk",
           "short", "unk", "none"}
MIN_GOLD_POS = 8          # positives AND negatives needed in gold to estimate reliability for a label
W_MIN, W_MAX = 0.2, 0.7


def script_of(text):
    ranges = {"latin": [(0x41, 0x24F)], "cyrillic": [(0x400, 0x4FF)], "cjk": [(0x4E00, 0x9FFF), (0x3040, 0x30FF)],
              "hangul": [(0xAC00, 0xD7AF)], "arabic": [(0x600, 0x6FF)], "greek": [(0x370, 0x3FF)]}
    cnt = {}
    for ch in str(text)[:1500]:
        o = ord(ch)
        for n, rs in ranges.items():
            if any(a <= o <= b for a, b in rs):
                cnt[n] = cnt.get(n, 0) + 1
                break
    return max(cnt, key=cnt.get) if cnt else "none"


def detect_lang(text):
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        return detect(str(text)[:1500]) if len(str(text).strip()) > 20 else "short"
    except Exception:
        return script_of(text)


def gold_mask(train):
    return train[LABELS].notna().all(axis=1)


def reliability_table(train, A, B):
    """AUC of A, B and their mean against gold labels; Cohen-style agreement A vs B on hard labels."""
    if "B_parse_ok" in B:
        B = B[B.B_parse_ok.astype(bool)]              # ignore placeholder rows B has not labeled yet
    A = A[A.StudyInstanceUID.isin(B.StudyInstanceUID)] if len(B) < len(A) else A
    g = train[gold_mask(train)].merge(A, on="StudyInstanceUID").merge(B, on="StudyInstanceUID")
    rows = []
    for l in LABELS:
        r = {"label": l, "n_gold": len(g)}
        if len(g):
            y = g[l].astype(int)
            r["n_pos"] = int(y.sum()); r["n_neg"] = int((1 - y).sum())
            if y.sum() >= MIN_GOLD_POS and (1 - y).sum() >= MIN_GOLD_POS:
                for k, col in (("auc_A", f"A_{l}"), ("auc_B", f"B_{l}")):
                    r[k] = roc_auc_score(y, g[col])
                r["auc_mean"] = roc_auc_score(y, (g[f"A_{l}"] + g[f"B_{l}"]) / 2)
        both = A[["StudyInstanceUID", f"A_{l}"]].merge(B[["StudyInstanceUID", f"B_{l}"]], on="StudyInstanceUID")
        hardA, hardB = both[f"A_{l}"] > .5, both[f"B_{l}"] > .5
        po = (hardA == hardB).mean()
        pe = hardA.mean() * hardB.mean() + (1 - hardA.mean()) * (1 - hardB.mean())
        r["kappa_A_vs_B_all"] = (po - pe) / (1 - pe) if pe < 1 else np.nan
        r["prev_A"], r["prev_B"] = float(hardA.mean()), float(hardB.mean())
        rows.append(r)
    return pd.DataFrame(rows).set_index("label")


DEFAULT_WA, DEFAULT_WB = 0.30, 0.50


def labeler_weights(rel):
    """Global mixing weights from gold AUC (AUC-0.5, floored). Two scalars, so little overfitting risk."""
    out = []
    for k, d in (("auc_A", DEFAULT_WA), ("auc_B", DEFAULT_WB)):
        v = rel[k].dropna().mean() if k in rel else np.nan
        out.append(max(float(v) - 0.5, 0.05) if v == v else d)
    return out[0], out[1]


def mix_label(df, l, uncovered, wa, wb):
    """-> (y, agree) for one label. Labeler A abstains when it found NO mention (gold shows its recall is low,
    so silence is not evidence of absence) or when the report language is not covered by its lexicon."""
    pa, ca = df[f"A_{l}"].fillna(.05).values, df[f"Ac_{l}"].fillna(0).values
    silent = df[f"An_{l}"].fillna(0).values == 0
    ca = np.where(uncovered | silent, 0.0, ca)
    pb = df[f"B_{l}"].fillna(.05).values
    cb = df[f"Bc_{l}"].fillna(0).values if f"Bc_{l}" in df else np.zeros(len(df))
    num, den = wa * ca * pa + wb * cb * pb, wa * ca + wb * cb
    y = np.where(den > 0, num / np.where(den > 0, den, 1), 0.05)
    agree = np.where((ca > 0) & (cb > 0), 1 - np.abs(pa - pb), 0.8)
    return y, agree


def combine(train, A, B):
    df = train[["StudyInstanceUID", "Report"]].copy()
    df["lang"] = df.Report.fillna("").map(detect_lang)
    df = df.merge(A, on="StudyInstanceUID", how="left").merge(B, on="StudyInstanceUID", how="left")
    gm = gold_mask(train).values
    rel = reliability_table(train, A, B)
    wa, wb = labeler_weights(rel)
    print(f"mixing weights: A={wa:.2f} B={wb:.2f}")
    out = pd.DataFrame({"StudyInstanceUID": train.StudyInstanceUID.values, "lang": df.lang.values,
                        "is_gold": gm})
    uncovered = ~df.lang.isin(A_LANGS).values
    print(f"Labeler A coverage: {int(uncovered.sum())} of {len(df)} reports are in languages A does not cover "
          f"-> A confidence set to 0 there. Languages: {df.lang[uncovered].value_counts().to_dict()}")
    for l in LABELS:
        y, agree = mix_label(df, l, uncovered, wa, wb)
        base = DEFAULT_BASE_W
        aucs = [rel.loc[l].get(k, np.nan) for k in ("auc_A", "auc_B", "auc_mean")]
        if not np.isnan(aucs[2]):
            base = float(np.clip(0.7 * max(0.0, (aucs[2] - 0.5) * 2), W_MIN, W_MAX))
        w = base * (0.5 + 0.5 * agree)
        goldv = train[l].values
        y = np.where(gm, goldv, y)
        w = np.where(gm, 1.0, w)
        partial = (~gm) & ~pd.isna(goldv)           # partially labeled: trust the provided label
        y = np.where(partial, goldv, y)
        w = np.where(partial, 1.0, w)
        out[f"y_{l}"] = y.astype(float)
        out[f"w_{l}"] = w.astype(float)
    return out, rel


def eval_gold(train, A, B):
    """Per-label gold AUC for A, B, plain mean, and the abstain-weighted mix used by `combine`."""
    rel = reliability_table(train, A, B)
    wa, wb = labeler_weights(rel)
    Bv = B[B.B_parse_ok.astype(bool)] if "B_parse_ok" in B else B
    g = train[gold_mask(train)].merge(A, on="StudyInstanceUID").merge(Bv, on="StudyInstanceUID")
    unc = np.zeros(len(g), bool)
    rows = []
    for l in LABELS:
        y = g[l].astype(int).values
        mixed, _ = mix_label(g, l, unc, wa, wb)
        r = {"label": l, "pos": int(y.sum()), "neg": int((1 - y).sum())}
        if 0 < y.sum() < len(y):
            r["AUC_A"] = roc_auc_score(y, g[f"A_{l}"]); r["AUC_B"] = roc_auc_score(y, g[f"B_{l}"])
            r["AUC_plain_mean"] = roc_auc_score(y, (g[f"A_{l}"] + g[f"B_{l}"]) / 2)
            r["AUC_mix"] = roc_auc_score(y, mixed)
        rows.append(r)
    res = pd.DataFrame(rows).set_index("label")
    return res, (wa, wb)


def make_audit_sheet(train, A, B, n_per_lang=40, seed=42, max_chars=1200):
    lang = train.Report.fillna("").map(detect_lang)
    d = train[["StudyInstanceUID", "Report"]].assign(lang=lang).merge(A, on="StudyInstanceUID").merge(B, on="StudyInstanceUID")
    parts = [g.sample(min(len(g), n_per_lang), random_state=seed) for _, g in d.groupby("lang")]
    s = pd.concat(parts).reset_index(drop=True)
    sheet = s[["StudyInstanceUID", "lang"]].copy()
    sheet["report"] = s.Report.astype(str).str.slice(0, max_chars)
    for l in LABELS:
        sheet[f"A_{l}"] = (s[f"A_{l}"] > .5).astype(int)
        sheet[f"B_{l}"] = (s[f"B_{l}"] > .5).astype(int)
    for l in LABELS:
        sheet[f"h_{l}"] = ""          # FILL IN: 1 = present, 0 = absent/not mentioned
    return sheet


def score_sheet(sheet):
    rows = []
    sheet = sheet.copy()
    for l in LABELS:
        h = pd.to_numeric(sheet[f"h_{l}"], errors="coerce")
        ok = h.notna()
        if ok.sum() == 0:
            continue
        for src in ("A", "B"):
            p = sheet.loc[ok, f"{src}_{l}"].astype(int); t = h[ok].astype(int)
            tp, fp, fn = int(((p == 1) & (t == 1)).sum()), int(((p == 1) & (t == 0)).sum()), int(((p == 0) & (t == 1)).sum())
            prec = tp / (tp + fp) if tp + fp else np.nan
            rec = tp / (tp + fn) if tp + fn else np.nan
            f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else np.nan
            rows.append(dict(label=l, labeler=src, n=int(ok.sum()), n_pos=int(t.sum()), precision=prec, recall=rec, f1=f1))
    per_label = pd.DataFrame(rows)
    per_lang = []
    for lg, g in sheet.groupby("lang"):
        for src in ("A", "B"):
            acc = []
            for l in LABELS:
                h = pd.to_numeric(g[f"h_{l}"], errors="coerce")
                ok = h.notna()
                if ok.sum():
                    acc.append((g.loc[ok, f"{src}_{l}"].astype(int) == h[ok].astype(int)).mean())
            if acc:
                per_lang.append(dict(lang=lg, labeler=src, n_reports=len(g), accuracy=float(np.mean(acc))))
    return per_label, pd.DataFrame(per_lang)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("combine"); c.add_argument("--train_csv"); c.add_argument("--a"); c.add_argument("--b")
    c.add_argument("--out"); c.add_argument("--reliability_out", default=None)
    au = sub.add_parser("audit"); au.add_argument("--train_csv"); au.add_argument("--a"); au.add_argument("--b")
    au.add_argument("--out"); au.add_argument("--n_per_lang", type=int, default=40)
    eg = sub.add_parser("evalgold"); eg.add_argument("--train_csv"); eg.add_argument("--a"); eg.add_argument("--b")
    sc = sub.add_parser("score"); sc.add_argument("--sheet"); sc.add_argument("--out_prefix", default="audit_score")
    a = ap.parse_args()
    if a.cmd == "combine":
        tr, A, B = pd.read_csv(a.train_csv), pd.read_csv(a.a), pd.read_csv(a.b)
        for l in LABELS:
            tr[l] = pd.to_numeric(tr[l], errors="coerce")
        out, rel = combine(tr, A, B)
        out.to_csv(a.out, index=False)
        print(rel.round(3).to_string())
        if a.reliability_out:
            rel.to_csv(a.reliability_out)
        print("gold studies:", int(out.is_gold.sum()), "| saved", a.out)
    elif a.cmd == "audit":
        tr, A, B = pd.read_csv(a.train_csv), pd.read_csv(a.a), pd.read_csv(a.b)
        make_audit_sheet(tr, A, B, a.n_per_lang).to_csv(a.out, index=False)
        print("wrote", a.out, "- fill the h_<label> columns (1/0) then run: score")
    elif a.cmd == "evalgold":
        tr = pd.read_csv(a.train_csv)
        for l in LABELS:
            tr[l] = pd.to_numeric(tr[l], errors="coerce")
        res, w = eval_gold(tr, pd.read_csv(a.a), pd.read_csv(a.b))
        print(res.round(2).to_string()); print("mean AUC:", res[["AUC_A", "AUC_B", "AUC_plain_mean", "AUC_mix"]].mean().round(3).to_dict())
        print(f"mixing weights used: A={w[0]:.2f} B={w[1]:.2f}  (58 gold studies -> differences under ~0.03 are noise)")
    else:
        pl, plg = score_sheet(pd.read_csv(a.sheet))
        pl.to_csv(a.out_prefix + "_per_label.csv", index=False); plg.to_csv(a.out_prefix + "_per_lang.csv", index=False)
        print(pl.round(3).to_string()); print(plg.round(3).to_string())
