import numpy as np
import zarr


def _create_array(group, name, shape, chunks, dtype):
    """Version-agnostic array creation: zarr v3 renamed Group.create_dataset
    to Group.create_array; this falls back to the v2 API if needed."""
    if name in group:
        return group[name]
    try:
        return group.create_array(name=name, shape=shape, chunks=chunks, dtype=dtype)
    except AttributeError:
        return group.create_dataset(name=name, shape=shape, chunks=chunks, dtype=dtype)


def _append_along_axis(arr, values, axis=0):
    """Version-agnostic append via explicit resize + slice assignment
    (works identically on zarr v2 and v3, unlike the old .append() method)."""
    old_shape = list(arr.shape)
    new_len = old_shape[axis] + values.shape[axis]
    new_shape = old_shape.copy()
    new_shape[axis] = new_len
    arr.resize(tuple(new_shape))

    slicer = [slice(None)] * len(new_shape)
    slicer[axis] = slice(old_shape[axis], new_len)
    arr[tuple(slicer)] = values


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
        "action_kernel",
        "trace_len",
    ]

    def __init__(
        self,
        path,
        n_v,
        n_c,
        n_p,
        n_stim,
        store_full_traces=False,
        chunk_trials=256,
        t_iti=None,
    ):
        self.store = zarr.open(path, mode="a")
        self.store_full_traces = store_full_traces
        self.t_iti = t_iti

        for col in self.SCALAR_FIELDS:
            _create_array(
                self.store, col, shape=(0,), chunks=(chunk_trials,), dtype="f4"
            )

        if store_full_traces:
            trace_specs = {
                "full_r_v": n_v,
                "full_r_c": n_c,
                "full_r_p": n_p,
                "full_x_stim": n_stim,
            }
            for key, n_units in trace_specs.items():
                _create_array(
                    self.store,
                    key,
                    shape=(0, 0, n_units),
                    chunks=(chunk_trials, 1, n_units),
                    dtype="f4",
                )

    def append(self, result, training_step):
        B = result.z_c.shape[0]
        rt_np = result.rt.cpu().numpy().astype("f4")
        trace_len = (
            (rt_np + 1 + self.t_iti)
            if self.t_iti is not None
            else np.full(B, np.nan, dtype="f4")
        )

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
            "rt": rt_np,
            "timed_out": result.timed_out.cpu().numpy().astype("f4"),
            "prev_model_choice": result.prev_model_choice.cpu().numpy().astype("f4"),
            "has_prev_choice": result.has_prev_choice.cpu().numpy().astype("f4"),
            "action_kernel": result.action_kernel.cpu().numpy().astype("f4"),
            "trace_len": trace_len,
        }
        for col, arr in scalars.items():
            _append_along_axis(self.store[col], arr, axis=0)

        if self.store_full_traces:
            self._append_trace("full_r_v", result.full_r_v)
            self._append_trace("full_r_c", result.full_r_c)
            self._append_trace("full_r_p", result.full_r_p)
            self._append_trace("full_x_stim", result.full_x_stim)

    def _append_trace(self, key, tensor):
        arr = tensor.detach().cpu().numpy().astype("f4")
        ds = self.store[key]
        if ds.shape[1] == 0:
            ds.resize((ds.shape[0], arr.shape[1], ds.shape[2]))
        _append_along_axis(ds, arr, axis=0)
