import sys

import torch
from torch.utils.data import TensorDataset, Dataset
from typing import Tuple, Sequence, Optional, List
import numpy as np
import h5py
from tqdm import tqdm_notebook as tqdm

from task.math import is_homogenous
from . import Run, Trace, prepare
from task.square import extract_from_trace


# Non sequential dataset
def generate_non_sequential_dataset(run: Run, number: int = 1, epsilon: float = 0.01, duration=10,
                                    keep=None, seed=None, disable_tqdm=False, name: str = '') -> Trace:
    traces = []
    size = 0

    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    assert keep is None or ((keep is not None) and 0 < keep < 1), "keep parameter should be a percentage ]0;1]"

    pbar = tqdm(total=int(number), disable=disable_tqdm)
    while True:
        tr = run(T=duration, epsilon=epsilon, seed=np.random.randint(0xDEADBEEF))
        if keep is not None:
            idx = np.round(np.linspace(0, len(tr.time) - 1, int(keep*len(tr.time)))).astype(int)
            tr = Trace(*[x[0][idx] for x in zip(tr)])
        traces.append(tr)
        tr_length = len(traces[-1].time)
        pbar.update(tr_length)
        size += tr_length
        if size > number:
            break
    pbar.close()

    trace = Trace(*[np.concatenate(x) for x in zip(*traces)])
    if name:
        if keep is not None:
            name += f"_{int(keep*100)}%"
        if seed is not None:
            name += f"_{seed}"
        with h5py.File(f"{name}.hdf5", "w") as f:
            for key, data in trace._asdict().items():
                f[key] = data
    return trace


def load_non_sequential_dataset(name: str = '') -> Trace:
    with h5py.File(f"{name}.hdf5", "r") as f:
        return Trace(**{key: data[:] for key, data in f.items()})


def central_dataset(trace: Trace, valid):
    N = trace.state.shape[1]
    tracestate = trace.state
    if is_homogenous(trace.state):
        tracestate = extract_from_trace(trace.state)

        x = tracestate.reshape(-1, N * 3)
        y = trace.control.reshape(-1, N * 2)
    else:
        x = tracestate.reshape(-1, N * 2)
        y = trace.control.reshape(-1, N * 2)

    train_cut = int((1 - valid) * len(x))
    train_dataset = (x[:train_cut], y[:train_cut])
    valid_dataset = (x[train_cut:len(x)], y[train_cut:len(x)])
    print(f"train: x {train_dataset[0].shape}\ty {train_dataset[1].shape}")
    print(f"valid: x {valid_dataset[0].shape}\ty {valid_dataset[1].shape}")
    return train_dataset, valid_dataset


def distributed_dataset(trace: Trace, valid):
    ss = trace.sensing[:, :, :, :2]
    ss = ss.reshape(ss.shape[0] * ss.shape[1], ss.shape[2] * ss.shape[3])

    cs = trace.control
    cs = cs.reshape(cs.shape[0] * cs.shape[1], cs.shape[2])

    train_cut = int((1 - valid) * len(ss))
    d_train_dataset = (ss[:train_cut], cs[:train_cut])
    d_valid_dataset = (ss[train_cut:len(ss)], cs[train_cut:len(ss)])
    print(f"x {d_train_dataset[0].shape}\ty {d_train_dataset[1].shape}")
    return d_train_dataset, d_valid_dataset


# Sequential dataset
def generate_sequential_dataset(run: Run,  number: int = 1,steps=None,  name: str = '',
                                duration: float = np.inf, epsilon: float = 0.01,
                                seed=None, keep=None, disable_tqdm=False) -> List[Trace]:
    traces = []
    size = 0

    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    assert keep is None or ((keep is not None) and 0 < keep < 1), "keep parameter should be a percentage ]0;1]"

    pbar = tqdm(total=int(number), disable=disable_tqdm)
    if steps is None:
        print("Undefined 'steps': assuming 'number' specifies the number of runs (instead of desired samples)")
        for _ in range(number):
            traces.append(run(T=duration, epsilon=epsilon, seed=np.random.randint(0xDEADBEEF)))
            pbar.update(1)
    else:
        while True:
            tr = run(T=duration, epsilon=epsilon, seed=np.random.randint(0xDEADBEEF))
            micro_traces = [Trace(*[prop[0][i:i + steps] for prop in zip(tr)])
                                 for i in np.arange(0, len(tr.state), steps) if i + steps < len(tr.state)]
            if keep is not None:
                idx = np.round(np.linspace(0, len(tr.time) - 1, int(keep * len(tr.time)))).astype(int)
                micro_traces = micro_traces[idx]
            traces.extend(micro_traces)
            new_samples = len(micro_traces)
            pbar.update(new_samples)
            size += new_samples
            if size > number:
                break
    pbar.close()

    if name:
        if steps is not None:
            name += f"_{steps}"
        if keep is not None:
            name += f"_{int(keep*100)}%"
        if seed is not None:
            name += f"_{seed}"
        with h5py.File(f"{name}.hdf5", "w") as f:
            for i, trace in enumerate(traces):
                g = f.create_group(f"run{i}")
                for key, data in trace._asdict().items():
                    g[key] = data
    return traces


def load_sequential_dataset(name: str = '') -> List[Trace]:
    with h5py.File(f"{name}.hdf5", "r") as f:
        return [Trace(**{key: data[:] for key, data in g.items()}) for g in f.values()]


def tensors_from_trace(trace: Trace) -> Tuple[torch.FloatTensor, ...]:
    return torch.FloatTensor(trace.sensing), torch.FloatTensor(trace.control)


class SequenceDataset(Dataset):

    def __init__(self, runs: Sequence[Tuple[torch.Tensor, ...]]) -> None:
        self._runs = runs

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        return self._runs[index]

    def __len__(self) -> int:
        return len(self._runs)


def sequence_dataset(traces: Sequence[Trace]) -> SequenceDataset:
    res = SequenceDataset([tensors_from_trace(trace) for trace in traces])
    #print(f"x {len(res)}")
    return res
