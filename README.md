# Knee MRI Abnormality Detection

Multimodal pipeline for the Kaggle **RSNA Knee Abnormality Detection** competition: predict 12 clinically important knee findings per MRI study from DICOM images, using the paired radiology reports as a (noisy) source of training labels.

> **Research code. Not a medical device, not validated for clinical use, and not medical advice.**

**Status:**  "Competition closed 22 Oct 2026. Final public score: 0.899, private score: 0/912, rank: 41292."`

---

## The task

- **Input:** one knee MRI study = several DICOM series (sagittal / coronal / axial), plus a free-text radiology report (training only).
- **Output:** a probability for each of 12 findings: `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`.
- **Metric:** macro-averaged ROC AUC over the 12 labels (only ranking matters; no thresholds or calibration needed for the score).
- **Hard part:** only **58 of 4,407** training studies have human labels. The rest have reports in about a dozen languages, so most of the work is turning reports into trustworthy labels and not overfitting to scanner or site.

## Approach

```
reports ──► Labeler A (multilingual rules)  ─┐
        └─► Labeler B (Qwen3-8B, few-shot) ──┴─► combine ──► soft labels + weights ─┐
                                                                                     ├─► train (site-grouped CV)
DICOMs ──► cache (series selection, per-series normalization, 24 slices @160px) ─────┘        │
                                                                                              ▼
                                                              fold ensemble ──► rank-average ──► submission.csv
```

1. **Labeler A – rules** (`labeler_rules.py`): lexicons for en/es/fr/de/pt/it/ru/zh with scope-aware negation ("no tear") and hedging ("cannot exclude") handling. High precision, low recall.
2. **Labeler B – LLM** (`labeler_llm.py`): Qwen3-8B (4-bit, run locally on a Kaggle GPU) extracts `present / absent / uncertain / not_mentioned` per finding as JSON. The data never leaves the notebook.
3. **Combine** (`labeler_combine.py`): A abstains when it finds no mention (its silence is not evidence of absence) and for languages its lexicon does not cover. Mixing weights come from agreement with the 58 gold studies. Gold labels override everything.
4. **Cache** (`cache_data.py`): per study, the best fluid-sensitive series per plane, sorted by physical position, 24 slices, percentile-clipped to uint8, padded to square, resized to 160×160.
5. **Model** (`model.py`): 2.5D slices → shared timm CNN → attention pooling over slices → transformer over series tokens (plane/fluid/fat embeddings) → 12 learned label queries cross-attend to the series, so each finding can choose its own planes.
6. **Training** (`train.py`): soft-target weighted BCE, AdamW, warmup + cosine, EMA, AMP, **5-fold GroupKFold by scanner (site proxy)**, OOF AUC with bootstrap CIs, per-site AUC.
7. **Inference** (`infer.py`): fold ensemble, rank averaging, fallback priors for unreadable studies, validated submission file.

## Key findings from the data (EDA)

- 4,407 training studies, 820k DICOM files, ~570 GB. Only 58 studies (1.3%) have all 12 gold labels, and that set is enriched (e.g. effusion 60%, fracture 31%), so it is not representative.
- Report language is tied to scanner: all 220 Cyrillic reports come from Philips, all 321 Greek reports from Siemens. A model can learn these shortcuts, hence site-grouped validation.
- Every study has all 3 planes and at least one fluid-sensitive, fat-suppressed series. `Fluid_Sensitive` and `Fat_Suppression` are identical in this data.
- The CSV plane column matched the DICOM orientation in 100% of 3,000 sampled series.
- Sampled DICOMs were all uncompressed (fast decode, ~8 ms/slice). The test set may contain compressed transfer syntaxes.
- Keyword matching on reports is unreliable: in the gold studies, a report mentioning "MCL" had a positive MCL label only 20% of the time (negation, normal findings).

Plots and tables are in [`eda_outputs/`](eda_outputs/).

## Results

### Report labelers vs. the 58 gold studies (AUC)

| | A (rules) | B (Qwen3-8B) | Plain mean | **Mix (used)** |
|---|---|---|---|---|
| Mean over 12 labels | 0.70 | 0.84 | 0.84 | **0.85** |

Per label (mix): ACL 0.96, MCL 0.94, Medial Meniscus 0.91, Lateral Meniscus 0.82, Medial OA 0.88, Lateral OA 0.86, PF OA 0.81, Effusion 0.77, Synovitis 0.76, Baker's 0.90, Contusion 0.77, Fracture 0.83.

With only 58 studies (9–35 positives per label), differences below ~0.05–0.1 are noise. Labeler B's prompt was **not** tuned on the gold studies, so these numbers are a fair estimate. Weakest labels: Effusion, Synovitis, Contusion, PF OA.

### Resulting training labels (4,407 studies, y ≥ 0.8)

ACL 650 · MCL 550 · Medial Meniscus 1,771 · Lateral Meniscus 705 · Medial OA 1,465 · Lateral OA 721 · PF OA 1,607 · Effusion 2,664 · Synovitis 1,018 · Baker's 1,308 · Contusion 641 · Fracture 229

### Image model

`TODO: fill in from /kaggle/working/models/cv_report.json — macro-AUC on report labels, macro-AUC on the 58 gold studies, per-label AUC with 95% CIs, worst-site AUC, training time per fold, and the leaderboard scores (public / private).`

| | CV (report labels) | CV (gold-58) | Public LB | Private LB |
|---|---|---|---|---|
| Baseline (EfficientNet-B0, 160px, 24 slices, K=1) | TODO | TODO | TODO | TODO |

## Repository layout

| Path | Contents |
|---|---|
| `knee_pipeline/` | The pipeline code (below) |
| `notebooks/` | `TODO: describe each notebook (EDA / labeling / caching / training / submission)` |
| `eda_outputs/` | Tables and plots exported from the EDA notebook |
| `refs/` | `TODO: describe (papers / notes)` |

`knee_pipeline/`:

| File | Purpose |
|---|---|
| `labeler_rules.py` | Labeler A: multilingual rules with negation/hedging |
| `labeler_llm.py` | Labeler B: LLM extraction (`hf` local, `anthropic`, `mock` backends; resumable cache) |
| `labeler_combine.py` | `combine`, `evalgold`, `audit`, `score` subcommands |
| `cache_data.py` | DICOM → uint8 arrays `[S, D, H, W]` |
| `model.py` | KneeNet, soft BCE loss, EMA |
| `train.py` | Site-grouped K-fold training and CV report |
| `infer.py` | Fold ensemble inference and submission writer |
| `tests/test_labeler_rules.py` | 54 multilingual checks for Labeler A |

## Reproducing (Kaggle)

The data (~570 GB) lives on Kaggle; run everything in Kaggle notebooks. Upload `knee_pipeline/` as a private dataset and attach the competition and a Qwen3-8B model from Kaggle Models.

```bash
CODE=/kaggle/input/<your-dataset>/knee_pipeline
DATA=/kaggle/input/competitions/rsna-knee-abnormality-detection
MODEL=/kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1

# 1) Labelers (Labeler B needs a GPU; ~6 s/report, ~8 h for all 4,407 reports; it resumes from its cache)
python $CODE/labeler_rules.py --train_csv $DATA/train.csv --out /kaggle/working/A.csv
python $CODE/labeler_llm.py   --train_csv $DATA/train.csv --out /kaggle/working/B.csv \
       --backend hf --model_path $MODEL --batch_size 4 --load_4bit

# 2) Check against gold, then build training labels
python $CODE/labeler_combine.py evalgold --train_csv $DATA/train.csv --a /kaggle/working/A.csv --b /kaggle/working/B.csv
python $CODE/labeler_combine.py combine  --train_csv $DATA/train.csv --a /kaggle/working/A.csv --b /kaggle/working/B.csv \
       --out /kaggle/working/labels_soft.csv --reliability_out /kaggle/working/reliability.csv

# 3) Cache images (CPU; ~9 GB at the defaults)
python $CODE/cache_data.py --split train --data_dir $DATA --out /kaggle/working/cache --D 24 --size 160 --K 1 --workers 4

# 4) Train (GPU)
python $CODE/train.py --cache_dir <cache> --labels <labels_soft.csv> --out /kaggle/working/models --epochs 20 --pretrained

# 5) Inference in the offline submission notebook (must use the same --D and --size as caching)
python $CODE/infer.py --models_dir <models> --data_dir $DATA --labels <labels_soft.csv> --out submission.csv --D 24 --size 160
```

Requirements: Python 3.10+, torch, timm, pydicom, pylibjpeg (+ `-libjpeg`, `-openjpeg`) or python-gdcm, pandas, numpy, scikit-learn, scipy, pillow. Optional: langdetect, transformers, bitsandbytes.

## Limitations

- **Labels are noisy.** They are derived from reports, scored against only 58 gold studies. Labeler lexicons were written from general knowledge and have **not** been reviewed by radiologists or native speakers; Greek and Cyrillic have only 3 gold studies each, so they can only be checked by hand-audit.
- **Effusion** is probably defined differently in the gold labels (clinically significant) than in reports (any mention). This is a hypothesis, not confirmed.
- **Validation is weak.** Site groups are scanner models, not hospitals. Report-label CV may not predict the leaderboard; other participants report CV-to-LB gaps of several points.
- `TODO: add anything you observed in the real training runs.`
- This is not a clinical tool. Matching a senior radiologist would need multi-reader, multi-site evaluation and a prospective study, not a leaderboard AUC.

## Data and licensing

The competition data and report text are **not** included and must not be redistributed; get them from the [competition page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) under its rules. Check what you commit: derived per-study files (labels, `study_meta.csv`, audit sheets containing report text) can fall under the data terms.

License: `TODO: choose one (e.g. MIT or Apache-2.0) and add a LICENSE file.`

## References

- Bien N. et al. *Deep-learning-assisted diagnosis for knee magnetic resonance imaging: development and retrospective validation of MRNet.* PLOS Medicine, 2018.
- Qiu et al. *CoPAS: multi-plane knee MRI abnormality detection.* Nature Communications, 2024.
- RSNA Knee Abnormality Detection, Kaggle, 2026.
- Models: Qwen3 (Alibaba Cloud, Apache-2.0).
