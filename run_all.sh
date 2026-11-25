#!/usr/bin/env bash
# Reproduce everything from scratch (about 2 minutes on a laptop CPU).
set -euo pipefail
cd "$(dirname "$0")"
pip install -r requirements.txt
python3 -I src/prepare_data.py
python3 -I src/train.py          # CV evaluation + conformal calibration + final model + VUS predictions
python3 -I src/sanity_checks.py  # label-shuffle and leave-gene-out checks
python3 -I src/plots.py          # figures -> reports/
echo "Done. Launch the demo with:  streamlit run app/app.py"
