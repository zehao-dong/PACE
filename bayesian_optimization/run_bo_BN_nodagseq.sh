#!/bin/bash

python bo_bn_nodagseq.py \
  --data-name="asia_200k" \
  --save-appendix=PACEnodagseq \
  --checkpoint=40 \
  --res-dir="BN_results_pace_nodagseq/" \
  --random-as-test \
  --random-baseline \
  --cuda_number 0 \

  #--save-appendix="SVAE" \
  #--save-appendix="GraphRNN" \
  #--save-appendix="GCN" \
  #--save-appendix="DeepGMG" \
