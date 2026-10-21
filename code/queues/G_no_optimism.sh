source "$(dirname "$0")/common.sh"
run run_experiments.py --dataset all --no-optimism-term --model-name qort_no_optimism --tag abl_no_optimism
