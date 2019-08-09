from typing import NamedTuple, List

import numpy as np
from abc import ABC, abstractmethod

class Task(ABC):


    @abstractmethod
    def initialize(self):
        ...

    @abstractmethod
    def update_targets(self, controller, super_controller) -> float:
        ...

    @property
    @abstractmethod
    def targets(self):
        ...

    @abstractmethod
    def distance(self, state) -> float:
        ...


class Trace(NamedTuple):
    time:           np.ndarray
    pos_state:      np.ndarray
    communication:  np.ndarray
    sensing:        np.ndarray
    control:        np.ndarray
    targets:        np.ndarray
    error:          np.ndarray


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
                # pad with (pos_state, communication, sensing, 0, targets, error)
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
        self.super_controller = controller
        self.dynamic = dynamic
        self.sensor = sensor
        self.dt = dt

    def __call__(self, epsilon: float = 0.01, T: float = np.inf) -> Trace:

        t = 0.0
        pos_state = self.task.initialize()
        self.controller = self.super_controller(self.task.targets)
        steps: List[Trace] = []
        error = self.task.distance
        e = error(pos_state)
        while (e > epsilon and t < T) or t == 0:

            sensing = self.sensor(pos_state)
            control, *communication = self.controller(pos_state, sensing)
            if communication:
                communication = communication[0]
            steps.append(Trace(t, pos_state, communication, sensing, control, self.task.targets, e))
            pos_state = self.dynamic(pos_state, control)
            self.controller = self.task.update_targets(self.controller, self.super_controller)
            t += self.dt
            e = error(pos_state)
        return Trace(*[np.array(x) for x in zip(*steps)])