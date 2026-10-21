source "$(dirname "$0")/common.sh"
run run_experiments.py --dataset all --single-prior --tag abl_single_prior
run run_experiments.py --dataset all --static-gate --tag abl_static_gate
run run_experiments.py --dataset all --no-relative-view --tag abl_no_relative_view
run run_experiments.py --dataset all --vibration-only --tag abl_vibration_only
run run_experiments.py --dataset all --lambda-vibration 0 --lambda-selector 0 --lambda-over 0 --lambda-gate 0 --model-name qort_no_aux --tag abl_no_aux
