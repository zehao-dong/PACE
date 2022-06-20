#!/usr/bin/env bash

for s in {1..10}
	do
		python bayesian_optimization/DNGO/dngo_ls_nasbench101.py --dim 64 --seed $s --init_size 16 --topk 5 --dataset nasbench101 --output_path dngo_ls_pace_nomask  --embedding_path pace_nasbench101_nomask.pt --computation_aware_search True
	done
