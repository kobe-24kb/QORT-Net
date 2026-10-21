source "$(dirname "$0")/common.sh"
for LR in 0.001 0.0005; do for DO in 0.10 0.20; do
  if [ "$LR" = "0.001" ] && [ "$DO" = "0.10" ]; then continue; fi
  run run_experiments.py --dataset both --learning-rate $LR --dropout $DO --model-name qort_lr${LR}_do${DO} --tag grid_qort_lr${LR}_do${DO}
done; done
run run_experiments.py --dataset both --train-subsets --tag subsets_qort
run run_experiments.py --dataset both --train-subsets --prior-only --tag subsets_prior_only
