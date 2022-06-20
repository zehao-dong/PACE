#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python train_pace_nas_nodagseq301.py --do_train --parallel --train_data data/nasbench301/train_data.pt --train_pair data/nasbench301/train_pair_k1_d5000000_metric_flops.pt  --valid_data data/nasbench301/test_data.pt --valid_pair data/nasbench301/test_pair_k1_d5000000_metric_flops.pt --dataset nasbench301 --n_vocab 11 --maxL 15 --batch_size 1024 --save_path pace_nodagseq
