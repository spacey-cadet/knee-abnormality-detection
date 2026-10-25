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
