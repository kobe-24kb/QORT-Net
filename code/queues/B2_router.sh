source "$(dirname "$0")/common.sh"
run run_experiments.py --dataset all --router-mode fixed --router-value 0.0 --model-name qort_router_r0 --tag router_r0
run run_experiments.py --dataset all --router-mode fixed --router-value 0.5 --model-name qort_router_r05 --tag router_r05
run run_experiments.py --dataset all --router-mode fixed --router-value 1.0 --model-name qort_router_r1 --tag router_r1
run run_experiments.py --dataset all --gate-base constant --gate-base-value 0.575 --model-name qort_gate_base_const --tag gate_base_const
