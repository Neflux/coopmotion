from gym.envs.registration import register

register(
    id='world-v0',
    entry_point='gym_world.envs:WorldEnv',
)
register(
    id='world-extrahard-v0',
    entry_point='gym_world.envs:WorldExtraHardEnv',
)