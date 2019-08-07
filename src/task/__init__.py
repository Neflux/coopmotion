from typing import NamedTuple, List

import numpy as np
from abc import ABC, abstractmethod


class Task(ABC):
    @abstractmethod
    def distance(self, state) -> float:
        ...

    @abstractmethod
    def initial(self):
        ...

    @property
    @abstractmethod
    def targets(self):
        ...


class Trace(NamedTuple):
    time: np.ndarray
    state: np.ndarray
    communication: np.ndarray
    sensing: np.ndarray
    control: np.ndarray
    targets: np.ndarray
    error: np.ndarray


def prepare(trace: Trace, steps = None, padding: bool = False,
            default_dt: float = 0.1
            ) -> Trace:
    if steps is not None:
        if len(trace.time) > steps:
            s = slice(steps)
            trace = Trace(*[x[s] for x in trace])
        elif len(trace.time) < steps:
            if padding:
                if len(trace.time) > 1:
                    dt = trace.time[-1] - trace.time[-2]
                else:
                    dt = default_dt
                items = [np.arange(0, dt * steps, dt)]
                # pad with (state, communication, sensing, 0, targets, error)
                n = steps - len(trace.time)
                for data, k in zip(trace[1:], [1, 1, 1, 0, 1, 1]):
                    last = [data[-1] * k for _ in range(n)]
                    items.append(np.concatenate([data, last]))
                trace = Trace(*items)
    return trace


class Run:

    def __init__(self, task: Task, controller, dynamic,
                 sensor, dt: float = 0.1):
        self.task = task
        self.macro_controller = controller
        self.dynamic = dynamic
        self.sensor = sensor
        self.dt = dt

    def __call__(self, epsilon: float = 0.01, T: float = np.inf) -> Trace:
        t = 0.0
        state = self.task.initial()
        # This is horrible, any other way?
        try:
            self.controller = self.macro_controller(self.task.targets)
        except:
            self.controller = self.macro_controller
        steps: List[Trace] = []
        error = self.task.distance
        e = error(state)
        while (e > epsilon and t < T) or t == 0:
            sensing = self.sensor(state)
            control, *communication = self.controller(state, sensing)
            if communication:
                communication = communication[0]
            steps.append(Trace(t, state, communication, sensing, control, self.task.targets, e))
            state = self.dynamic(state, control)
            t += self.dt
            e = error(state)
        trace = Trace(*[np.array(x) for x in zip(*steps)])
        return trace