#!/usr/bin/env bash

for s in {1..10}
	do
		python bayesian_optimization/DNGO/dngo_ls_nasbench301.py --dim 64 --seed $s --init_size 16 --topk 5 --dataset nasbench301 --output_path dngo_ls_dvae  --embedding_path dvae_nasbench301.pt --computation_aware_search True
	done
