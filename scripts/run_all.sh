#!/usr/bin/env bash
# Reproduce every experiment from scratch (about 6-8 CPU-hours on one core; runs in src/).
set -euo pipefail
cd "$(dirname "$0")/../src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 gen.py                                   # base-topology datasets (data_*.npz)
for s in 0 1 2; do
  for v in pigtfl gt_nophys gt_vanilla gcn cnn; do python3 train.py $v 25 $s; done
done
python3 train.py gt_nobias 25 0; python3 train.py gt_measonly 25 0
for v in pigtfl gt_nophys gt_nobias gt_vanilla; do python3 train.py $v 60 0 2000; done
python3 train.py cnn 100 0 2000
for s in 0 1 2; do python3 train.py pigtfl 200 $s 500; python3 train.py gt_vanilla 200 $s 500; done
python3 train.py gt_nophys 200 0 500; python3 train.py gt_nobias 200 0 500; python3 train.py cnn 300 0 500
python3 evaluate.py                              # Tables 3-4, Fig. 4 -> results.json
python3 gen_topo.py 0; python3 gen_topo.py 1     # 12 training + 4 unseen topologies
for v in pigtfl gt_nophys gt_nobias gt_vanilla gcn; do python3 train_topo.py $v 25 0; done
python3 train_topo.py pigtfl 25 1; python3 train_topo.py gt_vanilla 25 1
python3 eval_topo.py                             # Table 5, Fig. 5 -> results_topo_train.json
python3 conf_eval.py; python3 conf_eval2.py      # Table 6 (Algorithm 2)
python3 fig_journal.py                           # Fig1-Fig5 (.png, .tiff)
