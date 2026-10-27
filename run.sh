#!/usr/bin/env bash
# run.sh
set -euo pipefail


python -m modularRNN.train \
  training.device=cpu training.use_wandb=true training.compile_model=false \
  model.n_visp=32 model.n_mos=32 model.rank_ff=3 model.n_fb=2  model.fb_mode="zero"\
  env.batch_size=32 \
  env.block_len_mean=60 env.block_len_min=20 env.block_len_max=100 \
  env.T_max=100 env.T_iti=3 \
  decision.t_min=10 decision.theta=0.9 \
  env.kappa_kernel=0.5 env.xi_kernel=0.2 \
  training.n_steps=25001 \
  training.chunk_dt_start=150 training.chunk_dt_growth_every=500 training.chunk_dt_max=1000 \
  training.ckpt_every=1000 \
  "$@"


  # xi_kernel: 1.0
  # kappa_kernel: 0.9  
  # this is if we want only previous choice.

  # reward timing and kernel timing should also change to pulse and iti
  # Wmv zeroed from the start (architectural ablation, trained without it)
  # python -m modularRNN.train model.fb_mode=zero ...

  # Wmv sparse, 10% of entries trainable
  # python -m modularRNN.train model.fb_mode=sparse model.fb_sparsity=0.1 ...

  # previous-choice-only signal instead of an EMA trace
  # python -m modularRNN.train env.xi_kernel=1.0 env.kappa_kernel=0.9 model.n_fb=2 ...

  # reward persists through ITI, kernel is pulse-only (decoupled)
  # python -m modularRNN.train env.reward_timing=persist_iti env.kernel_timing=pulse_only env.kappa_kernel=0.5 model.n_fb=2 ...