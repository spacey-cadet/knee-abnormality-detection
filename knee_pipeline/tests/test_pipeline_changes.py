"""Tests for the v4 changes (run: python tests/test_pipeline_changes.py from the repo root)."""
import json, os, sys, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import LABELS
import labeler_llm as L
import labeler_combine as C


def test_parse_failure_gets_zero_confidence_and_is_retried():
    class Bad:
        calls = 0
        def generate(self, reports):
            Bad.calls += 1
            return ["not json"] * len(reports) if Bad.calls == 1 else [json.dumps({l: "present" for l in LABELS})] * len(reports)
    df = pd.DataFrame({"StudyInstanceUID": ["a", "b"], "Report": ["x", "y"]})
    cache = tempfile.mktemp(suffix=".jsonl")
    r1 = L.run(df, Bad(), cache)
    assert (~r1.B_parse_ok).all() and (r1[[f"Bc_{l}" for l in LABELS]].values == 0).all()   # was 0.4 before the fix
    r2 = L.run(df, Bad(), cache)                      # second run retries failed rows from the cache
    assert r2.B_parse_ok.all() and (r2["Bv_ACL"] == "present").all()


def test_prompt_versions():
    assert L.PROMPTS["v1"] != L.PROMPTS["v2"] and L.PROMPTS["v2"].startswith(L.PROMPTS["v1"])


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    train = pd.DataFrame({"StudyInstanceUID": [f"s{i}" for i in range(n)], "Report": ["tear of the ACL. effusion." for _ in range(n)]})
    A = train[["StudyInstanceUID"]].copy(); B = train[["StudyInstanceUID"]].copy(); B["B_parse_ok"] = True
    for l in LABELS:
        train[l] = np.nan
        train.loc[:59, l] = rng.integers(0, 2, 60).astype(float)
        v = np.where(rng.random(n) < .5, "present", "not_mentioned")
        B[f"Bv_{l}"] = v; B[f"B_{l}"] = np.where(v == "present", .95, .05); B[f"Bc_{l}"] = np.where(v == "present", .85, .4)
        A[f"A_{l}"] = .05; A[f"Ac_{l}"] = .4; A[f"An_{l}"] = 0
    return train, A, B


def test_calibration_moves_present_prob_toward_gold_rate():
    train, A, B = _toy()
    table, pooled = C.fit_verdict_probs(train, B)
    assert all(0.01 <= table[l][v] <= 0.99 for l in LABELS for v in C.VERDICTS)
    assert pooled["present"] < 0.95                 # random gold => ~0.5, not the hard-coded 0.95
    out, _ = C.combine(train, A, B, calibrate_b=True)
    assert out.loc[60:, "y_ACL"].max() < 0.9


def test_audit_blind_full_text_weighted_and_scored():
    train, A, B = _toy()
    train["Report"] = ["long report " * 200 + f" impression {i}" for i in range(len(train))]
    sheet, key = C.make_audit_sheet(train, A, B, n_per_lang=20)
    assert not any(c.startswith(("A_", "B_", "Bv_")) for c in sheet.columns)          # blind
    assert sheet.report.str.len().max() > 1200 and sheet.report.str.contains("impression").all()  # not truncated
    assert (key[key.stratum == "gold"].weight == 1).all()
    for l in LABELS:                                   # simulate a perfect annotator = B
        sheet[f"h_{l}"] = key.set_index("StudyInstanceUID").loc[sheet.StudyInstanceUID, f"B_{l}"].values
    pl, plg = C.score_sheet(sheet, key)
    assert (pl[pl.labeler == "B"].f1.dropna() > 0.99).all()


def test_b_only_equals_calibrated_b_and_ignores_a():
    train, A, B = _toy()
    for l in LABELS:                                  # make A loud and different so it WOULD matter in the mix
        A[f"A_{l}"] = .95; A[f"Ac_{l}"] = .9; A[f"An_{l}"] = 3
    mix, _ = C.combine(train, A, B, calibrate_b=True)
    bo, _ = C.combine(train, A, B, calibrate_b=True, b_only=True)
    table, _ = C.fit_verdict_probs(train, B)
    exp = B[f"Bv_ACL"].map(table["ACL"]).values[60:]
    assert np.allclose(bo.loc[60:, "y_ACL"].values, exp, atol=1e-6)      # y == calibrated B probability
    assert not np.allclose(mix.loc[60:, "y_ACL"].values, exp, atol=1e-3)  # while the mix is moved by A
    assert (bo.loc[:59, [f"y_{l}" for l in LABELS]].values == train.loc[:59, LABELS].values).all()  # gold untouched


def test_eval_labels_are_fixed_and_fold_checked():
    from common import attach_eval_labels, check_folds
    n = 10
    ids = [f"s{i}" for i in range(n)]
    cols = {f"y_{l}": np.linspace(0, 1, n) for l in LABELS}
    df = pd.DataFrame({"StudyInstanceUID": ids, **{k: v * 0.7 for k, v in cols.items()}})      # v4-like targets (max 0.7)
    old = pd.DataFrame({"StudyInstanceUID": ids, "fold": [i % 5 for i in range(n)], **cols})
    p = tempfile.mktemp(suffix=".csv"); old.to_csv(p, index=False)
    out = attach_eval_labels(df.copy(), p)
    assert np.allclose(out["ev_ACL"], cols["y_ACL"]) and out["y_ACL"].max() < 0.8 <= out["ev_ACL"].max()
    out["fold"] = old["fold"].values; assert check_folds(out) == 0
    out["fold"] = (out["fold"] + 1) % 5
    try:
        check_folds(out); raise AssertionError("should have raised")
    except ValueError:
        pass
    try:                                              # a study missing from the eval file must not pass silently
        old.iloc[:5].to_csv(p, index=False); attach_eval_labels(df.copy(), p); raise AssertionError("should have raised")
    except ValueError:
        pass
    same = attach_eval_labels(df.copy(), None)         # default: ev == y
    assert np.allclose(same["ev_ACL"], same["y_ACL"])


def test_a_labels_mixes_only_the_named_labels():
    train, A, B = _toy()
    for l in LABELS:
        A[f"A_{l}"] = .95; A[f"Ac_{l}"] = .9; A[f"An_{l}"] = 3
    mix, _ = C.combine(train, A, B, calibrate_b=True)
    bo, _ = C.combine(train, A, B, calibrate_b=True, b_only=True)
    sel, _ = C.combine(train, A, B, calibrate_b=True, a_labels=["MCL", "Fracture"])
    for l in LABELS:
        want = mix if l in ("MCL", "Fracture") else bo
        assert np.allclose(sel[f"y_{l}"], want[f"y_{l}"]) and np.allclose(sel[f"w_{l}"], want[f"w_{l}"]), l
    try:
        C.combine(train, A, B, a_labels=["NotALabel"]); raise AssertionError("should have raised")
    except ValueError:
        pass


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
