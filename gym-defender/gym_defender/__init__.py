from gym.envs.registration import register

register(
    id='defender',
    entry_point='gym_defender.envs:Defender'
    #,kwargs={'K':20, 'initial_potential':0.8}
)

