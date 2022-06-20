#!/bin/bash

export PYTHONPATH="$(pwd)"

python bo_na.py \
  --data-name final_structures6 \
  --save-appendix PACE \
  --checkpoint 100 \
  --res-dir="ENAS_results_pace/" \
  --cuda_number 0 \
  --random-baseline \
  --random-as-test \
  #--save-appendix SVAE \
  #--save-appendix GraphRNN \
  #--save-appendix GCN \
  #--save-appendix DeepGMG \
  #--save-appendix DVAE_fast \

