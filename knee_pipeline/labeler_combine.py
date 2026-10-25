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


VERDICTS = ["present", "absent", "uncertain", "not_mentioned"]


def fit_verdict_probs(train, B, strength=10.0):
    """Empirical P(gold=1 | B verdict) per label, shrunk toward the pooled rate for that verdict.

    labeler_llm.py hard-codes present=0.95 / absent=0.03 / not_mentioned=0.05 / uncertain=0.5. On the 58 gold
    studies the real rates were ~0.73 / 0.02 / 0.12 / 0.29: 'present' is overconfident, which makes soft labels
    too hard. `strength` = pseudo-counts of the pooled prior (10 => a label needs >>10 cases to move far from it).
    """
    Bg = B[B.B_parse_ok.astype(bool)] if "B_parse_ok" in B else B
    g = train[gold_mask(train)].merge(Bg, on="StudyInstanceUID")
    pooled = {}
    for v in VERDICTS:
        ys = np.concatenate([g.loc[g[f"Bv_{l}"] == v, l].astype(float).values for l in LABELS]) if len(g) else np.array([])
        pooled[v] = (ys.sum() + 1) / (len(ys) + 2) if len(ys) else {"present": .9, "absent": .03, "uncertain": .5, "not_mentioned": .05}[v]
    table = {}
    for l in LABELS:
        table[l] = {}
        for v in VERDICTS:
            m = (g[f"Bv_{l}"] == v) if len(g) else np.array([], bool)
            n, k = int(m.sum()), float(g.loc[m, l].astype(float).sum()) if len(g) else 0.0
            table[l][v] = float(np.clip((k + strength * pooled[v]) / (n + strength), 0.01, 0.99))
    return table, pooled


def apply_verdict_probs(B, table):
    """Overwrite B_<label> with the calibrated probability for each row's verdict (confidences untouched)."""
    B = B.copy()
    for l in LABELS:
        B[f"B_{l}"] = B[f"Bv_{l}"].map(table[l]).fillna(B[f"B_{l}"]).values
    return B


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


def combine(train, A, B, calibrate_b=False, b_only=False, a_labels=None):
    if calibrate_b:
        table, pooled = fit_verdict_probs(train, B)
        print("calibrated P(gold | verdict), pooled:", {k: round(v, 2) for k, v in pooled.items()})
        B = apply_verdict_probs(B, table)
    df = train[["StudyInstanceUID", "Report"]].copy()
    df["lang"] = df.Report.fillna("").map(detect_lang)
    df = df.merge(A, on="StudyInstanceUID", how="left").merge(B, on="StudyInstanceUID", how="left")
    gm = gold_mask(train).values
    rel = reliability_table(train, A, B)
    wa, wb = labeler_weights(rel)
    # Which labels use Labeler A at all: None = all (A+B mix); set() = none (B only); set of names = only those.
    if a_labels is not None:
        bad = [x for x in a_labels if x not in LABELS]
        if bad:
            raise ValueError(f"unknown label(s) in a_labels: {bad}")
        a_set = set(a_labels)
    else:
        a_set = set() if b_only else None
    print(f"mixing weights: A={wa:.2f} B={wb:.2f} | A+B mix on: "
          + ("all labels" if a_set is None else (", ".join(sorted(a_set)) or "no labels (B only)")))
    out = pd.DataFrame({"StudyInstanceUID": train.StudyInstanceUID.values, "lang": df.lang.values,
                        "is_gold": gm})
    uncovered = ~df.lang.isin(A_LANGS).values
    print(f"Labeler A coverage: {int(uncovered.sum())} of {len(df)} reports are in languages A does not cover "
          f"-> A confidence set to 0 there. Languages: {df.lang[uncovered].value_counts().to_dict()}")
    for l in LABELS:
        bo = a_set is not None and l not in a_set     # B only for this label: A weight 0, agreement constant 0.8
        y, agree = mix_label(df, l, np.ones(len(df), bool) if bo else uncovered, 0.0 if bo else wa, wb)
        base = DEFAULT_BASE_W
        aucs = [rel.loc[l].get(k, np.nan) for k in ("auc_A", "auc_B", "auc_mean")]
        k_auc = 1 if bo else 2                      # B alone vs the A+B mean
        if not np.isnan(aucs[k_auc]):
            base = float(np.clip(0.7 * max(0.0, (aucs[k_auc] - 0.5) * 2), W_MIN, W_MAX))
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


def _ll(y, p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def eval_gold_softlabels(train, A, B, folds=5, seed=0):
    """Does calibrating B's verdict probabilities improve the SOFT labels the CNN trains on?
    Reports log-loss vs gold (lower is better) for B and the A+B mix, current lookup vs verdict probabilities
    fitted on the other folds (cross-validated by report, so no leakage). AUC cannot see this: it ignores calibration."""
    Bv = B[B.B_parse_ok.astype(bool)] if "B_parse_ok" in B else B
    rel = reliability_table(train, A, B)
    wa, wb = labeler_weights(rel)
    gold = train[gold_mask(train)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(gold)) % folds
    rows = []
    for k in range(folds):
        tr_g, te_g = gold[fold != k], gold[fold == k]
        table, _ = fit_verdict_probs(tr_g.assign(), B)       # fit on training folds only
        te = te_g.merge(A, on="StudyInstanceUID").merge(Bv, on="StudyInstanceUID")
        cal = apply_verdict_probs(Bv[Bv.StudyInstanceUID.isin(te.StudyInstanceUID)], table)
        te_cal = te_g.merge(A, on="StudyInstanceUID").merge(cal, on="StudyInstanceUID")
        unc = np.zeros(len(te), bool)
        for l in LABELS:
            y = te[l].astype(float).values
            mix_now, _ = mix_label(te, l, unc, wa, wb)
            mix_cal, _ = mix_label(te_cal, l, unc, wa, wb)
            rows.append(dict(label=l, n=len(te), LL_B_now=_ll(y, te[f"B_{l}"].values), LL_B_cal=_ll(y, te_cal[f"B_{l}"].values),
                             LL_mix_now=_ll(y, mix_now), LL_mix_cal=_ll(y, mix_cal)))
    d = pd.DataFrame(rows).groupby("label").apply(lambda g: pd.Series({c: np.average(g[c], weights=g.n) for c in
                                                   ["LL_B_now", "LL_B_cal", "LL_mix_now", "LL_mix_cal"]}))
    return d.loc[LABELS]


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


def make_audit_sheet(train, A, B, n_per_lang=40, seed=42, max_chars=None, disagree_frac=0.5, double_frac=0.2):
    """Blind, full-text, stratified audit sample.

    Returns (sheet, key). Give annotators `sheet` only (report text + empty h_* columns, rows shuffled);
    keep `key` (A/B labels, stratum, inverse-probability weight, double_label flag) for `score`.
    Strata per language: gold studies (always included) + A/B-disagreement studies (>= 2 labels differ,
    oversampled: they are where errors live) + random others. Weight = pool size / n sampled, so scores
    computed with the weights estimate the full population despite the oversampling.
    max_chars=None keeps the whole report (the old 1200-char cut hid the impression section).
    """
    lang = train.Report.fillna("").map(detect_lang)
    d = train[["StudyInstanceUID", "Report"]].assign(lang=lang, is_gold=gold_mask(train).values)
    d = d.merge(A, on="StudyInstanceUID").merge(B, on="StudyInstanceUID")
    d["n_disagree"] = sum(((d[f"A_{l}"] > .5) != (d[f"B_{l}"] > .5)).astype(int) for l in LABELS)
    rng = np.random.default_rng(seed)
    picked = []
    for lg, g in d.groupby("lang"):
        gold_g, rest = g[g.is_gold], g[~g.is_gold]
        picked += [(u, lg, "gold", 1.0) for u in gold_g.StudyInstanceUID]
        k = max(n_per_lang - len(gold_g), 0)
        pools = {"disagree": (rest[rest.n_disagree >= 2], int(round(k * disagree_frac))),
                 "random": (rest[rest.n_disagree < 2], k - int(round(k * disagree_frac)))}
        for name, (pool, kk) in pools.items():
            kk = min(kk, len(pool))
            if kk:
                idx = rng.choice(len(pool), kk, replace=False)
                picked += [(u, lg, name, len(pool) / kk) for u in pool.StudyInstanceUID.values[idx]]
    key = pd.DataFrame(picked, columns=["StudyInstanceUID", "lang", "stratum", "weight"])
    key["double_label"] = rng.random(len(key)) < double_frac
    ds = d.set_index("StudyInstanceUID")
    for l in LABELS:
        key[f"A_{l}"] = (ds.loc[key.StudyInstanceUID, f"A_{l}"].values > .5).astype(int)
        key[f"B_{l}"] = (ds.loc[key.StudyInstanceUID, f"B_{l}"].values > .5).astype(int)
        key[f"Bv_{l}"] = ds.loc[key.StudyInstanceUID, f"Bv_{l}"].values
    rep = ds.loc[key.StudyInstanceUID, "Report"].astype(str)
    sheet = key[["StudyInstanceUID", "lang"]].copy()
    sheet["report"] = (rep.str.slice(0, max_chars) if max_chars else rep).values
    for l in LABELS:
        sheet[f"h_{l}"] = ""          # FILL IN: 1 = present, 0 = absent/not mentioned, blank = skip
    order = rng.permutation(len(sheet))       # shuffle: row order must not reveal the stratum
    return sheet.iloc[order].reset_index(drop=True), key


def _wcounts(p, t, w):
    return tuple(float(w[(p == pp) & (t == tt)].sum()) for pp, tt in ((1, 1), (1, 0), (0, 1), (0, 0)))


def _wmetrics(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    po = (tp + tn) / n if n else np.nan
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2 if n else np.nan
    return prec, rec, f1, ((po - pe) / (1 - pe) if n and pe != 1 else np.nan)


def score_sheet(sheet, key=None):
    """Score A and B against human labels. With `key` (from make_audit_sheet): labels come from the key and
    every case is inverse-probability weighted. Without it: the old behaviour (A_/B_ columns inside the sheet)."""
    sheet = sheet.copy()
    if key is not None:
        k = key.set_index("StudyInstanceUID")
        sheet = sheet.set_index("StudyInstanceUID")
        for c in [c for c in k.columns if c[:2] in ("A_", "B_") or c == "weight"]:
            sheet[c] = k.loc[sheet.index, c]
        sheet = sheet.reset_index()
    w_all = sheet["weight"].values if "weight" in sheet else np.ones(len(sheet))
    rows = []
    for l in LABELS:
        h = pd.to_numeric(sheet[f"h_{l}"], errors="coerce")
        ok = h.notna().values
        if ok.sum() == 0:
            continue
        for src in ("A", "B"):
            p = sheet.loc[ok, f"{src}_{l}"].astype(int).values; t = h[ok].astype(int).values
            prec, rec, f1, kap = _wmetrics(*_wcounts(p, t, w_all[ok]))
            rows.append(dict(label=l, labeler=src, n=int(ok.sum()), n_pos=int(t.sum()), precision=prec, recall=rec, f1=f1, kappa=kap))
    per_label = pd.DataFrame(rows)
    per_lang = []
    for lg, g in sheet.groupby("lang"):
        w_g = g["weight"].values if "weight" in g else np.ones(len(g))
        for src in ("A", "B"):
            p_all, t_all, ww = [], [], []
            for l in LABELS:
                h = pd.to_numeric(g[f"h_{l}"], errors="coerce"); ok = h.notna().values
                if ok.sum():
                    p_all.append(g.loc[ok, f"{src}_{l}"].astype(int).values); t_all.append(h[ok].astype(int).values); ww.append(w_g[ok])
            if p_all:
                p, t, ww = np.concatenate(p_all), np.concatenate(t_all), np.concatenate(ww)
                _, _, f1, kap = _wmetrics(*_wcounts(p, t, ww))
                per_lang.append(dict(lang=lg, labeler=src, n_reports=len(g), accuracy=float((ww * (p == t)).sum() / ww.sum()), f1=f1, kappa=kap))
    return per_label, pd.DataFrame(per_lang)


def interannotator(sheet1, sheet2):
    """Cohen's kappa between two annotators over cells both filled in (the ceiling any labeler can reach)."""
    a, b = sheet1.set_index("StudyInstanceUID"), sheet2.set_index("StudyInstanceUID")
    ids = a.index.intersection(b.index)
    y1 = np.concatenate([pd.to_numeric(a.loc[ids, f"h_{l}"], errors="coerce").values for l in LABELS])
    y2 = np.concatenate([pd.to_numeric(b.loc[ids, f"h_{l}"], errors="coerce").values for l in LABELS])
    ok = ~np.isnan(y1) & ~np.isnan(y2)
    return _wmetrics(*_wcounts(y1[ok].astype(int), y2[ok].astype(int), np.ones(ok.sum())))[3], int(ok.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("combine"); c.add_argument("--train_csv"); c.add_argument("--a"); c.add_argument("--b")
    c.add_argument("--out"); c.add_argument("--reliability_out", default=None)
    c.add_argument("--calibrate_b", action="store_true", help="map B verdicts to gold-calibrated probabilities")
    c.add_argument("--a_labels", default=None, help="comma-separated labels that keep the A+B mix; all others use B only "
                   "(e.g. 'MCL,Lateral Meniscus,Fracture')")
    c.add_argument("--b_only", action="store_true", help="ignore Labeler A entirely (gold AUC: B alone beat the A+B mix on 9/12 labels)")
    au = sub.add_parser("audit"); au.add_argument("--train_csv"); au.add_argument("--a"); au.add_argument("--b")
    au.add_argument("--out"); au.add_argument("--n_per_lang", type=int, default=40)
    au.add_argument("--max_chars", type=int, default=None, help="default: keep the full report")
    eg = sub.add_parser("evalgold"); eg.add_argument("--train_csv"); eg.add_argument("--a"); eg.add_argument("--b")
    sc = sub.add_parser("score"); sc.add_argument("--sheet"); sc.add_argument("--out_prefix", default="audit_score")
    sc.add_argument("--key", default=None, help="<out>_key.csv written by `audit` (needed for blind sheets)")
    sc.add_argument("--sheet2", default=None, help="second annotator's filled sheet (inter-annotator kappa)")
    a = ap.parse_args()
    if a.cmd == "combine":
        tr, A, B = pd.read_csv(a.train_csv), pd.read_csv(a.a), pd.read_csv(a.b)
        for l in LABELS:
            tr[l] = pd.to_numeric(tr[l], errors="coerce")
        out, rel = combine(tr, A, B, calibrate_b=a.calibrate_b, b_only=a.b_only,
                           a_labels=[s.strip() for s in a.a_labels.split(',')] if a.a_labels else None)
        out.to_csv(a.out, index=False)
        print(rel.round(3).to_string())
        if a.reliability_out:
            rel.to_csv(a.reliability_out)
        print("gold studies:", int(out.is_gold.sum()), "| saved", a.out)
    elif a.cmd == "audit":
        tr, A, B = pd.read_csv(a.train_csv), pd.read_csv(a.a), pd.read_csv(a.b)
        for l in LABELS:
            tr[l] = pd.to_numeric(tr[l], errors="coerce")
        sheet, key = make_audit_sheet(tr, A, B, a.n_per_lang, max_chars=a.max_chars)
        kp = a.out.rsplit(".", 1)[0] + "_key.csv"
        sheet.to_csv(a.out, index=False); key.to_csv(kp, index=False)
        sheet[sheet.StudyInstanceUID.isin(key[key.double_label].StudyInstanceUID)].to_csv(a.out.rsplit(".", 1)[0] + "_2.csv", index=False)
        print(key.groupby(["lang", "stratum"]).size().unstack(fill_value=0).to_string())
        print("wrote", a.out, "(blind, give to annotators) +", kp, "(keep) + _2.csv (second annotator subset)")
        print("then fill h_<label> (1/0) and run: score --sheet <filled> --key", kp)
    elif a.cmd == "evalgold":
        tr = pd.read_csv(a.train_csv)
        for l in LABELS:
            tr[l] = pd.to_numeric(tr[l], errors="coerce")
        res, w = eval_gold(tr, pd.read_csv(a.a), pd.read_csv(a.b))
        print(res.round(2).to_string()); print("mean AUC:", res[["AUC_A", "AUC_B", "AUC_plain_mean", "AUC_mix"]].mean().round(3).to_dict())
        print(f"mixing weights used: A={w[0]:.2f} B={w[1]:.2f}  (58 gold studies -> differences under ~0.03 are noise)")
        sl = eval_gold_softlabels(tr, pd.read_csv(a.a), pd.read_csv(a.b))
        print("\nSoft-label log-loss vs gold (5-fold CV by report; lower = better):"); print(sl.round(3).to_string())
        print("mean:", sl.mean().round(3).to_dict())
    else:
        key = pd.read_csv(a.key) if a.key else None
        pl, plg = score_sheet(pd.read_csv(a.sheet), key)
        if a.sheet2:
            kap, n = interannotator(pd.read_csv(a.sheet), pd.read_csv(a.sheet2))
            print(f"inter-annotator kappa = {kap:.2f} over {n} cells (ceiling for any labeler)")
        pl.to_csv(a.out_prefix + "_per_label.csv", index=False); plg.to_csv(a.out_prefix + "_per_lang.csv", index=False)
        print(pl.round(3).to_string()); print(plg.round(3).to_string())
