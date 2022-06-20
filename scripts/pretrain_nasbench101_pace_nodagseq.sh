#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python train_pace_nas_nodagseq.py --do_train --parallel --train_data data/nasbench101/train_data.pt --train_pair data/nasbench101/train_pair_k2_d2000000_metric_params.pt  --valid_data data/nasbench101/test_data.pt --valid_pair data/nasbench101/test_pair_k2_d2000000_metric_params.pt --dataset nasbench101 --n_vocab 5 --maxL 7 --save_path pace_nodagseq/
