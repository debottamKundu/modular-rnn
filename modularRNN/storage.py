import numpy as np
import zarr


class HiddenStateStore:
    SCALAR_FIELDS = [
        "training_step",
        "session_id",
        "block_id",
        "trial_in_block",
        "block_bias",
        "contrast",
        "stim_side",
        "correct_label",
        "choice",
        "reward",
        "rt",
        "timed_out",
        "prev_model_choice",
        "has_prev_choice",
    ]

    def __init__(
        self, path, n_visp, n_mos, n_stim, store_full_traces=False, chunk_trials=256
    ):
        self.store = zarr.open(path, mode="a")
        self.store_full_traces = store_full_traces
        for col in self.SCALAR_FIELDS:
            if col not in self.store:
                self.store.create_dataset(  # type: ignore
                    col, shape=(0,), chunks=(chunk_trials,), dtype="f4"
                )

        if store_full_traces:
            trace_specs = {"full_r_v": n_visp, "full_r_m": n_mos, "full_x_stim": n_stim}
            for key, n_units in trace_specs.items():
                if key not in self.store:
                    self.store.create_dataset(  # type: ignore
                        key,
                        shape=(0, 0, n_units),
                        chunks=(chunk_trials, 1, n_units),
                        dtype="f4",
                    )

    def append(self, result, training_step):
        B = result.z_c.shape[0]
        scalars = {
            "training_step": np.full(B, training_step, dtype="f4"),
            "session_id": result.session_id.cpu().numpy().astype("f4"),
            "block_id": result.block_id.cpu().numpy().astype("f4"),
            "trial_in_block": result.trial_in_block.cpu().numpy().astype("f4"),
            "block_bias": result.block_bias.cpu().numpy().astype("f4"),
            "contrast": result.contrast.cpu().numpy().astype("f4"),
            "stim_side": result.stim_side.cpu().numpy().astype("f4"),
            "correct_label": result.correct_label.cpu().numpy().astype("f4"),
            "choice": result.choice.cpu().numpy().astype("f4"),
            "reward": result.reward.cpu().numpy().astype("f4"),
            "rt": result.rt.cpu().numpy().astype("f4"),
            "timed_out": result.timed_out.cpu().numpy().astype("f4"),
            "prev_model_choice": result.prev_model_choice.cpu().numpy().astype("f4"),
            "has_prev_choice": result.has_prev_choice.cpu().numpy().astype("f4"),
        }
        for col, arr in scalars.items():
            self.store[col].append(arr)  # type: ignore

        if self.store_full_traces:
            self._append_trace("full_r_v", result.full_r_v)
            self._append_trace("full_r_m", result.full_r_m)
            self._append_trace("full_x_stim", result.full_x_stim)

    def _append_trace(self, key, tensor):
        arr = tensor.detach().cpu().numpy().astype("f4")
        ds = self.store[key]
        if ds.shape[1] == 0:  # type: ignore
            ds.resize((ds.shape[0], arr.shape[1], ds.shape[2]))  # type: ignore
        ds.append(arr, axis=0)  # type: ignore
