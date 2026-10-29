#!/usr/bin/env bash
# run.sh
set -euo pipefail

python -m modularRNN.resume training.resume_from=./model/checkpoints/fc8dzrdu/ckpt_final_10000.pt training.n_steps=20001 training.run_id="fc8dzrdu"

# total steps is +old ones