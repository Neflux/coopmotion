from typing import NamedTuple, List, Callable

import numpy as np
from abc import ABC, abstractmethod

from task.square import h_dynamic, nh_dynamic


class Task(ABC):
    target_xys: np.ndarray
    initial: Callable
    holonomic: bool

    @abstractmethod
    def h(self):
        ...

    @abstractmethod
    def nh(self):
        ...

    @abstractmethod
    def update(self):
        ...

    @abstractmethod
    def distance(self):
        ...


class Trace(NamedTuple):
    time:           np.ndarray
    state:          np.ndarray
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
                # pad with (state, communication, sensing, 0, targets, error)
                n = steps - len(trace.time)
                for data, k in zip(trace[1:], [1, 1, 1, 0, 1, 1]):
                    last = [data[-1] * k for _ in range(n)]
                    items.append(np.concatenate([data, last]))
                trace = Trace(*items)
    return trace


class Run:

    def __init__(self, task: Task, controller,
                 sensor, dt: float = 0.1):
        self.task = task
        self.super_controller = controller
        self.dynamic = h_dynamic(dt) if task.holonomic else nh_dynamic(dt)
        self.sensor = sensor
        self.dt = dt

    def __call__(self, epsilon: float = 0.01, T: float = np.inf) -> Trace:

        state, targets = self.task.initial()
        controller = self.super_controller(targets)
        error = self.task.distance()
        update = self.task.update()

        t = 0.0
        steps: List[Trace] = []
        e = error(state)
        while (e > epsilon and t < T) or t == 0:
            # Acquire sensing info from simulation state
            sensing = self.sensor(state)
            # Acquire control decision from state/sensing, depending on the controller
            control, *communication = controller(state, sensing)
            if communication:
                communication = communication[0]
            steps.append(Trace(t, state, communication, sensing, control, self.task.target_xys, e))
            # Update simulation state
            state = self.dynamic(state, control)
            # Optional task update of targets and therefore optimal controllers
            controller, self.task.target_xys = update(controller, self.super_controller, state)
            t += self.dt
            e = error(state)
        return Trace(*[np.array(x) for x in zip(*steps)])