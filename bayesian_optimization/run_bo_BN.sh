#!/bin/bash

python bo_bn.py \
  --data-name="asia_200k" \
  --save-appendix= PACE \
  --checkpoint=40 \
  --res-dir="BN_mask_pace/" \
  --random-as-test \
  --random-baseline \
  --cuda_number 0 \

  #--save-appendix="SVAE" \
  #--save-appendix="GraphRNN" \
  #--save-appendix="GCN" \
  #--save-appendix="DeepGMG" \
