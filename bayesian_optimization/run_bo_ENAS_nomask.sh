#!/bin/bash

export PYTHONPATH="$(pwd)"

python bo_na_nomask.py \
  --data-name final_structures6 \
  --save-appendix PACEnomask \
  --checkpoint 100 \
  --res-dir="ENAS_results_pace_nomask/" \
  --BO-rounds 10 \
  --BO-batch-size 50 \
  --cuda_number 0 \
  --random-baseline \
  --random-as-test \
  #--save-appendix SVAE \
  #--save-appendix GraphRNN \
  #--save-appendix GCN \
  #--save-appendix DeepGMG \
  #--save-appendix DVAE_fast \

