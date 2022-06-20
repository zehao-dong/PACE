#!/usr/bin/env bash

for s in {1..10}
	do
		python bayesian_optimization/DNGO/dngo_ls_nasbench301.py --dim 64 --seed $s --init_size 16 --topk 5 --dataset nasbench301 --output_path dngo_ls_pace_nomask  --embedding_path pace_nasbench301_nomask.pt --computation_aware_search True
	done
