source "$(dirname "$0")/common.sh"
run run_targeted_baselines.py --dataset all --models gru,film,filternet,patchtst,pca_drsn_tcn,mixlinear,vpnet --tag main_vibration --save-checkpoints
run run_targeted_baselines.py --dataset all --models gru,film --information-equated --window 40 --epochs 120 --patience 18 --plateau-patience 5 --lambda-over 0.05 --model-suffix _budget --tag budget_equated
for LR in 0.001 0.0005; do for DO in 0.15 0.25; do
  if [ "$LR" = "0.001" ] && [ "$DO" = "0.15" ]; then continue; fi
  run run_targeted_baselines.py --dataset both --models gru,film --information-equated --learning-rate $LR --dropout $DO --model-suffix _lr${LR}_do${DO} --tag grid_equated_lr${LR}_do${DO}
done; done
