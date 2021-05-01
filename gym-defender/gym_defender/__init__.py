from gym.envs.registration import register

register(
    id='defender-v0',
    entry_point='gym_defender.envs:Defender'
)

