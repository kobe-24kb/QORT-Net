source "$(dirname "$0")/common.sh"
run run_experiments.py --dataset all --token-mode no_spectral --model-name qort_no_spectral --tag abl_no_spectral
