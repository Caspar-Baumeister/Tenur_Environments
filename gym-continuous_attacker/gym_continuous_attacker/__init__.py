from gym.envs.registration import register

register(
    id='continuous_attacker-v0',
    entry_point='gym_continuous_attacker.envs:CAttacker'
)

