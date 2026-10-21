source "$(dirname "$0")/common.sh"
run run_experiments.py --dataset all --token-mode no_trend --model-name qort_no_trend --tag abl_no_trend
