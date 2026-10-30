#!/usr/bin/env bash
# run.sh
set -euo pipefail


# python -m modularRNN.train \
#   training.device=cpu training.use_wandb=true training.compile_model=false \
#   model.n_visp=32 model.n_mos=32 model.rank_ff=3 model.n_fb=2  model.fb_mode="sparse" model.stim_fb_gain=0.1 model.notes=["Sparse feedback matrix, ak and feedback are pulsed, ak is previous choice"]\
#   env.batch_size=32 \
#   env.block_len_mean=60 env.block_len_min=20 env.block_len_max=100 \
#   env.T_max=100 env.T_iti=5 \
#   decision.t_min=10 decision.theta=0.9 \
#   env.kappa_kernel=0.9 env.xi_kernel=1.0 \
#   training.n_steps=10001 \
#   training.chunk_dt_start=150 training.chunk_dt_growth_every=500 training.chunk_dt_max=1000 \
#   training.ckpt_every=2000 \
#   "$@"


# python -m modularRNN.train \
#   training.device=cpu training.use_wandb=true training.compile_model=false \
#   model.n_visp=32 model.n_mos=32 model.rank_ff=3 model.n_fb=1  model.fb_mode="dense" model.stim_fb_gain=0.1 model.notes=["Dense feedback matrix, longer ITI, no feedback"]\
#   env.batch_size=32 \
#   env.kappa_kernel=0 \
#   env.block_len_mean=60 env.block_len_min=20 env.block_len_max=100 \
#   env.T_max=120 env.T_iti=20 \
#   env.reward_timing="pulse_only" env.kernel_timing="persist_iti" \
#   decision.t_min=10 decision.theta=0.85 \
#   training.n_steps=25001 \
#   training.chunk_dt_start=150 training.chunk_dt_growth_every=500 training.chunk_dt_max=3000 \
#   training.ckpt_every=5000 \
#   "$@"

# maybe we can taper off the kernel input.

python -m modularRNN.train \
  training.device=cpu training.use_wandb=True training.compile_model=false \
  training.n_steps=20001 training.enable_early_stopping=false \
  model.n_v=32 model.n_c=32 model.n_p=64 \
  model.n_stim=2 model.n_fb=2 \
  model.tau_v_range="[20.0,60.0]" model.tau_c_range="[60.0,80.0]" model.tau_p_range="[100.0,120.0]" \
  model.conn_c_from_v.mode=lowrank model.conn_c_from_v.rank=5 model.conn_c_from_v.gain=0.4 \
  model.conn_v_from_c.mode=lowrank model.conn_v_from_c.rank=5 model.conn_v_from_c.gain=0.4 \
  model.conn_p_from_v.mode=zero \
  model.conn_v_from_p.mode=dense model.conn_v_from_p.gain=0.4 \
  model.conn_p_from_c.mode=lowrank model.conn_p_from_c.rank=5 model.conn_p_from_c.gain=0.4 \
  model.conn_c_from_p.mode=dense model.conn_c_from_p.gain=0.4 \
  env.batch_size=32 env.n_blocks=6 env.n_unbiased_blocks=1 \
  env.block_len_mean=60 env.block_len_min=20 env.block_len_max=100 \
  env.T_max=120 env.T_iti=20 \
  env.reward_timing="pulse_only" env.kernel_timing="persist_iti" \
  env.kappa_kernel=0.5 env.xi_kernel=0.2 \
  decision.t_min=15 decision.theta=0.9 \
  training.chunk_dt_start=150 training.chunk_dt_growth_every=500 training.chunk_dt_max=3000 \
  training.ckpt_every=5000 \



  # reward timing and kernel timing should also change to pulse and iti
  # Wmv zeroed from the start (architectural ablation, trained without it)
  # python -m modularRNN.train model.fb_mode=zero ...

  # Wmv sparse, 10% of entries trainable
  # python -m modularRNN.train model.fb_mode=sparse model.fb_sparsity=0.1 ...

  # previous-choice-only signal instead of an EMA trace
  # python -m modularRNN.train env.xi_kernel=1.0 env.kappa_kernel=0.9 model.n_fb=2 ...

  # reward persists through ITI, kernel is pulse-only (decoupled)
  # python -m modularRNN.train env.reward_timing=persist_iti env.kernel_timing=pulse_only env.kappa_kernel=0.5 model.n_fb=2 ...