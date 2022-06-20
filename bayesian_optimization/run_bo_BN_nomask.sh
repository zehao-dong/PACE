#!/bin/bash

python bo_bn_nomask.py \
  --data-name="asia_200k" \
  --save-appendix=PACEnomask \
  --checkpoint=50 \
  --res-dir="BN_results_pace_nomask/" \
  --random-as-test \
  --random-baseline \

  #--save-appendix="SVAE" \
  #--save-appendix="GraphRNN" \
  #--save-appendix="GCN" \
  #--save-appendix="DeepGMG" \
