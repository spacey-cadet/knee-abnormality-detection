# Knee MRI pipeline (RSNA knee abnormality challenge)

Implements the spec: Labeler A (rules) + Labeler B (LLM) -> soft labels -> cached DICOM -> KneeNet -> site-grouped CV -> inference.

## Files
| File | Purpose |
|---|---|
| `labeler_rules.py` | Labeler A: multilingual (en es fr de pt it ru zh) rules, scope-aware negation/hedging |
| `labeler_llm.py` | Labeler B: LLM JSON extraction. Backends: `hf` (local, recommended), `anthropic` (API), `mock` |
| `labeler_combine.py` | `combine` (soft labels + weights + reliability), `audit` (sheet to hand-check), `score` (score your audit) |
| `cache_data.py` | DICOM -> uint8 arrays [S,D,H,W]; per-series normalization; series selection by plane/fluid/fat |
| `model.py` | KneeNet (2.5D CNN, slice attention, series transformer, 12 label queries), SoftBCE, EMA |
| `train.py` | Site-grouped K-fold, AMP, EMA, OOF AUC with bootstrap CI, per-site AUC, optional gold fine-tune |
| `infer.py` | Fold ensemble, rank averaging, fallback priors, validated submission.csv |
| `run_pipeline_example.sh` | Command order |
| `tests/test_labeler_rules.py` | 54 multilingual checks for Labeler A |

## Requirements
python 3.10+, torch, timm, pydicom, pylibjpeg + pylibjpeg-libjpeg + pylibjpeg-openjpeg (or python-gdcm) for JPEG Lossless / JPEG 2000,
pandas, numpy, scikit-learn, scipy, pillow. Optional: langdetect, transformers (Labeler B hf), anthropic (Labeler B API).
Kaggle code competitions score offline: attach packages, pretrained weights (`--weights_path`) and your trained fold*.pt as Kaggle Datasets.

## Before you trust anything
1. Run the EDA notebook first; check gold-label count, languages, decode speed.
2. Labeler lexicons are NOT radiologist/native-speaker reviewed. Run `audit` -> fill `h_*` -> `score`, per language.
3. `anthropic` backend sends competition reports to a third party. Check the rules. Prefer `hf`.
4. Tested only on a tiny synthetic dataset (CPU). Not run on the real data: expect to tune paths, memory, batch size.
5. Disk: uint8 cache = studies*S*D*H*W bytes (5000*3*24*160*160 = 9.2 GB). Keep it under the ~20 GB working disk.
6. Check the competition's runtime limit and time `infer.py` on 100 studies first.

## v4 changes (labeling quality)
Tested: `PYTHONPATH=. python tests/test_pipeline_changes.py` and `tests/test_labeler_rules.py` (54/54). The LLM prompt v2 is NOT tested
on a real model (no GPU here): compare v1 vs v2 on a gold dev split before adopting it.

| File | Change | Why (evidence from the 58 gold studies) |
|---|---|---|
| `labeler_llm.py` | Failed parses now get confidence 0 (was 0.4, i.e. silently "not_mentioned") | bug; had no effect yet only because all 4,407 parsed |
| `labeler_llm.py` | Failed parses are re-labelled automatically (`--no_retry_failed` to disable) | replaces the notebook cache-filter cell |
| `labeler_llm.py` | `--prompt v1|v2` (default v1 = unchanged) | v2 adds rules for MCL vs lateral collateral, bone-marrow edema, mild/hedged findings, contradictions |
| `labeler_llm.py` | `--temperature`, `--seed`; writes `<out>.meta.json` | K sampled runs -> vote-fraction confidence; reproducibility |
| `labeler_combine.py` | `combine --calibrate_b`: B verdicts -> gold-calibrated P(gold) (shrunk to pooled rate) | measured rates present 0.73 / absent 0.02 / not_mentioned 0.12 / uncertain 0.29 vs hard-coded 0.95 / 0.03 / 0.05 / 0.50 |
| `labeler_combine.py` | `evalgold` also prints soft-label log-loss (5-fold CV by report) | AUC ignores calibration; the CNN trains on the probabilities |
| `labeler_combine.py` | `audit`: blind (no A/B shown), full report text, gold always included, disagreements oversampled, inverse-probability weights, `_key.csv`, `_2.csv` for a second annotator | old sheet cut reports at 1200 chars, showed A/B, sampled 7/360 gold |
| `labeler_combine.py` | `score --key ... [--sheet2 ...]`: weighted P/R/F1/kappa per label and language, inter-annotator kappa | |

Measured effect of `--calibrate_b` on gold (cross-validated log-loss of the soft label, lower is better): mix 0.512 -> 0.417, B alone 0.521 -> 0.409.
Note B alone (calibrated) is as good as the A+B mix on these 58 studies, so A adds little here; recheck on the audit labels.

New commands:
```
python labeler_llm.py --train_csv $D/train.csv --out $W/B.csv --backend hf --model_path $M --load_4bit --prompt v2
python labeler_llm.py ... --out $W/B_s1.csv --temperature 0.6 --seed 1      # repeat for seeds 1..5
python labeler_combine.py audit   --train_csv $D/train.csv --a $W/A.csv --b $W/B.csv --out $W/audit_sheet.csv
python labeler_combine.py score   --sheet $W/audit_filled.csv --key $W/audit_sheet_key.csv --sheet2 $W/audit_filled_2.csv
python labeler_combine.py combine --train_csv $D/train.csv --a $W/A.csv --b $W/B.csv --out $W/labels_soft.csv --calibrate_b
```

## v5 changes (training on improved labels, same 160 px EfficientNet-B0)
Tested without torch/GPU: `PYTHONPATH=. python tests/test_pipeline_changes.py` (label + eval-label logic). `train.py` was only
syntax-checked, NOT run: smoke-test it before a long job.

| File | Change | Why |
|---|---|---|
| `train.py` | `--eval_labels <csv>`: validation, best-epoch choice, early stopping and the CV report use FIXED labels (`ev_<label>`); training still uses `--labels`. Errors out if a study is missing or the fold split differs from the file's `fold` column. `oof.csv` `y_<label>` = eval labels, new `ytrain_<label>` = training targets | v4 calibrated labels put "present" at 0.73, below the 0.8 positive threshold, so validation would have lost nearly all positives |
| `train.py` | If every epoch scores NaN, keep the final EMA weights instead of crashing | `best_state` stayed `None` |
| `common.py` | `attach_eval_labels`, `check_folds` (pandas only) | testable without torch |
| `labeler_combine.py` | `combine --a_labels "MCL,Lateral Meniscus,Fracture"`: only the named labels keep the A+B mix, all others use B only (selected on the same 58 gold studies, so a weak choice) | the A+B mix beat B alone by 0.014 / 0.029 / 0.018 gold AUC on these three, and lost on the rest |
| `labeler_combine.py` | `combine --b_only`: Labeler A weight 0, per-label weight from B's own gold AUC | gold AUC: B alone beat the A+B mix on 9 of 12 labels |

```
python labeler_combine.py combine --train_csv $D/train.csv --a $W/A.csv --b $W/B.csv --out $W/labels_soft_v5b.csv --calibrate_b --b_only
python train.py --cache_dir $CACHE160 --labels $W/labels_soft_v5b.csv --eval_labels <baseline oof_all.csv> --out $W/exp_v5b \
    --only_folds 0 --epochs 12 --pretrained --workers 2 --train_slices 16 --bs 4 --accum 2     # default --seed 42 = baseline
```
