#!/usr/bin/env bash
# Order of operations (Kaggle: run each as "!python ..." in a notebook cell). Adjust paths.
set -e
W=/kaggle/working; D=${KNEE_DATA_DIR:-/kaggle/input/YOUR-COMPETITION-FOLDER}
export KNEE_DATA_DIR=$D PYTHONPATH=$PWD
python labeler_rules.py --train_csv $D/train.csv --out $W/A.csv                         # Labeler A
python labeler_llm.py   --train_csv $D/train.csv --out $W/B.csv --backend hf --model_path /kaggle/input/YOUR-LLM  # Labeler B
python labeler_combine.py audit   --train_csv $D/train.csv --a $W/A.csv --b $W/B.csv --out $W/audit_sheet.csv      # fill h_* columns by hand
python labeler_combine.py combine --train_csv $D/train.csv --a $W/A.csv --b $W/B.csv --out $W/labels_soft.csv --reliability_out $W/reliability.csv
python cache_data.py --split train --out $W/cache --D 24 --size 160 --K 1
python train.py --cache_dir $W/cache --labels $W/labels_soft.csv --out $W/models --epochs 20 --pretrained
python infer.py --models_dir $W/models --labels $W/labels_soft.csv --out $W/submission.csv
