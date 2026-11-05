#!/usr/bin/env bash
# Regenerate the datasets from fixed seeds and evaluate the released weights (about 30-45 min).
set -euo pipefail
cd "$(dirname "$0")/../src"
export OMP_NUM_THREADS=2
cp ../weights/*.pkl .
python3 gen.py
python3 evaluate.py
python3 conf_eval2.py
python3 gen_topo.py 0; python3 gen_topo.py 1
python3 eval_topo.py
python3 fig_journal.py
