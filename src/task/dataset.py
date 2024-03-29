import torch
from torch.utils.data import TensorDataset, Dataset
from typing import Tuple, Sequence, Optional, List
import numpy as np
import h5py

from . import Run, Trace, prepare


# Non sequential dataset
def generate_randomstart_dataset(run: Run, number: int = 1) -> Trace:
    traces = [run(T=0, epsilon=0.01) for i in range(number)]
    return Trace(*[np.concatenate(x) for x in zip(*traces)])

# Non sequential dataset
def generate_non_sequential_dataset(run: Run, number: int = 1, epsilon: float = 0.01, name: str = '') -> Trace:
    # why T=0?
    traces = [run(T=np.inf, epsilon=epsilon) for i in range(number)]

    trace = [np.concatenate(x) for x in zip(*traces)]
    reindex = np.arange(len(trace[0]))
    np.random.shuffle(reindex)
    trace = [x[reindex] for x in trace]
    trace = Trace(*trace)
    if name:
        with h5py.File(f"{name}.hdf5", "w") as f:
            for key, data in trace._asdict().items():
                f[key] = data
    return trace


def load_non_sequential_dataset(name: str = '') -> Trace:
    with h5py.File(f"{name}.hdf5", "r") as f:
        return Trace(**{key: data[:] for key, data in f.items()})


def central_dataset(trace: Trace) -> TensorDataset:
    return TensorDataset(torch.FloatTensor(trace.state).flatten(start_dim=1), torch.FloatTensor(trace.control).flatten(start_dim=1))


def distributed_dataset(trace: Trace) -> TensorDataset:
    ss = trace.sensing
    cs = trace.control
    return TensorDataset(
        torch.FloatTensor(ss.reshape(ss.shape[0]*ss.shape[1], ss.shape[2]*ss.shape[3])),
        torch.FloatTensor(cs.reshape(cs.shape[0]*cs.shape[1], cs.shape[2]))
    )


# Sequential dataset
def generate_sequential_dataset(run: Run, number: int = 1, name: str = '',
                                duration: float = np.inf, epsilon: float = 0.01) -> List[Trace]:
    traces = [run(epsilon=epsilon, T=duration) for i in range(number)]
    if name:
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


def sequence_dataset(traces: Sequence[Trace], steps: Optional[int] = None, padding: bool = False
                     ) -> SequenceDataset:
    return SequenceDataset([tensors_from_trace(prepare(trace, steps=steps, padding=padding))
                            for trace in traces])
