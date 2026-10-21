source "$(dirname "$0")/common.sh"
run run_trivial_baselines.py --dataset xjtu,phm,xjtu_c1,phm_c2,phm60 --tag trivial
run run_experiments.py --dataset all --tag main_qort --save-checkpoints
run run_experiments.py --dataset all --prior-only --tag main_prior_only --save-checkpoints
run run_targeted_baselines.py --dataset all --models gru,film,patchtst --information-equated --tag main_equated --save-checkpoints
run run_experiments.py --dataset phm60 --tag phm60_qort --save-checkpoints
run run_experiments.py --dataset phm60 --prior-only --tag phm60_prior_only
run run_targeted_baselines.py --dataset phm60 --models gru,film --information-equated --tag phm60_equated
run run_experiments.py --dataset both --window 20 --short-window 10 --model-name qort_w20_10 --tag window_20_10
run run_experiments.py --dataset both --window 80 --short-window 40 --model-name qort_w80_40 --tag window_80_40
