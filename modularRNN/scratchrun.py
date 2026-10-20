import torch
from model import ModularCTRNN
from env import ClosedLoopEnv
from generators import SessionPlanGenerator
from storage import HiddenStateStore

device = torch.device("cpu")
model = ModularCTRNN(
    n_visp=8,
    n_mos=8,
    n_stim=2,
    n_fb=1,
    rank_ff=2,
    dt=10.0,
    tau_visp_range=(20, 60),
    tau_mos_range=(100, 400),
    learnable_tau=False,
    rec_gain=1.2,
    ff_gain=0.4,
    fb_gain=0.4,
    in_gain=1.0,
    sigma_baseline=0.05,
)
plan_gen = SessionPlanGenerator(
    n_blocks=2,
    n_unbiased_blocks=1,
    bias_high=0.8,
    block_len_mean=10,
    block_len_min=5,
    block_len_max=15,
    possible_contrasts=[0.0, 0.25, 1.0],
    device=device,
)
env = ClosedLoopEnv(
    model,
    plan_gen,
    batch_size=2,
    device=device,
    T_stim=10,
    T_resp=15,
    T_iti=5,
    t_min=3,
    theta=0.5,
    sigma_stim=1.0,
    hold_fb_during_stim=True,
)
env.reset()
store = HiddenStateStore("/tmp/smoke_test.zarr", 8, 8, 2, store_full_traces=True)
for _ in range(10):
    res = env.run_trial(store_full_trace=True)
    store.append(res, training_step=0)

print(store.store["full_r_v"].shape)
print(store.store["session_id"][:])


"""
command:
python modularRNN/train.py experiment=server_run \
  model.n_visp=64 model.n_mos=64 model.rank_ff=3 \
  env.n_blocks=5 env.n_unbiased_blocks=1 \
  env.block_len_mean=60 env.block_len_min=20 env.block_len_max=100 \
  env.contrasts="[0.0,0.0625,0.125,0.25,1.0]" \
  training.n_steps=50000 \
  training.k_start=4 training.k_max=60 training.k_growth_every=500 \
  training.eval_every=200 training.probe_every=1000 training.ckpt_every=1000
  """