from gym.envs.registration import register

register(
    id='defender-v0',
    entry_point='gym_defender.envs:Defender'
    #,kwargs={'K':20, 'initial_potential':0.8}
)

print("works")

