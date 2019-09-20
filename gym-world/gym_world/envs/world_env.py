from typing import List

import gym
from gym import spaces
from scipy.spatial import distance

from task import Task, Trace
import numpy as np


class WorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self, task: Task, sensor, dt: float = 0.1):
        super(WorldEnv, self).__init__()

        self.task = task
        self.N = task.N
        self.error = task.distance()
        self.update_targets = task.update()
        from task.square import h_dynamic
        self.dt = dt
        self.dynamic = h_dynamic(dt)
        self.sensor = sensor(holonomic=True)

        self.collision_threshold = 0.01

        self.viewer = None
        self.world_length = 400

        high = 0.1
        self.action_space = spaces.Box(low=-high, high=high, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=self.sensor.get_shape(self.N)[1:], dtype=np.float32)

    def step(self, action_n):
        self.state = self.dynamic(self.state, action_n)
        self.update_targets(None, lambda x: None, self.state)

        done, info = False, 'default'
        reward = np.log(np.linalg.norm(self.task.target_xys - np.array(self.state), axis=1)) + 1

        sorted_dist = np.sort(distance.cdist(self.state, self.state, 'euclidean') + np.eye(self.N) * np.sqrt(2), axis=1)
        colliding_robots = np.any(sorted_dist < self.collision_threshold, axis=1)
        reward[colliding_robots] = -10
        if np.any(colliding_robots):
            done, info = True, "robot-robot collision"

        wall_dist = np.abs(self.state - np.clip(self.state, self.collision_threshold, 1. - self.collision_threshold))
        outer_robots = np.any(wall_dist > 0, axis=1)
        reward[outer_robots] = -10
        if np.any(outer_robots):
            done, info = True, "robot-wall collision"

        return self.sensor(self.state), reward, done, info

    def reset(self):
        self.state, self.targets = self.task.initial()
        return self.sensor(self.state)

    def render(self, mode='human', close=False):


        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(self.world_length, self.world_length)
            self.robottrans = []
            self.targettrans = []
            for i in range(self.N):
                robot = rendering.make_circle(6)
                robot.set_color(.5, .5, .8)
                self.robottrans.append(rendering.Transform())
                robot.add_attr(self.robottrans[-1])
                self.viewer.add_geom(robot)

                target = rendering.make_circle(6)
                target.set_color(.9, .1, .1)
                self.targettrans.append(rendering.Transform())
                target.add_attr(self.targettrans[-1])
                self.viewer.add_geom(target)

        if self.state is None: return None

        state = self.state * self.world_length
        targets = self.task.target_xys * self.world_length
        for i in range(self.N):
            self.robottrans[i].set_translation(state[i, 0], state[i, 1])
            self.targettrans[i].set_translation(targets[i, 0], targets[i, 1])
            # self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
