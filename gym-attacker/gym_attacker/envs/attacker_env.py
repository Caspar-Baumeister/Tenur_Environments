import gym
from gym import error, spaces, utils
from gym.utils import seeding
from random import randint, choice
import numpy as np


###### GYM ######

class Attacker(gym.Env):


    def __init__(self, K, initial_potential, verbose=0, random_probability=0.5):
        self.state = None
        self.K = K
        self.initial_potential = initial_potential
        self.weights = np.power(2.0, [-(self.K - i) for i in range(self.K + 1)])
        self.done = 0
        self.reward = 0
        self.action_space = spaces.Discrete(self.K)
        self.observation_space= spaces.MultiDiscrete([10]* (K+1))
        self.geo_prob = .3
        self.unif_prob = .4
        self.diverse_prob = .3
        self.high_one_prob = 0.2
        self.geo_high = self.K - 2
        self.unif_high = max(3, self.K-3)
        self.geo_ps = [0.45, 0.5, 0.6, 0.7, 0.8]
        self.verbose= verbose
        self.random_probability = random_probability        

    def potential(self, A):
        return np.sum(A*self.weights)


    def split(self, A):
        B = np.array([z - a for z, a in zip(self.state, A)])
        return A, B


    def erase(self, A):
        """Function to remove the partition A from the game state
        Arguments:
            A {list} -- The list representing the partition we want to remove
        """

        self.state = [z - a for z, a in zip(self.state, A)]
        self.state = np.array([0] + self.state[:-1])


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def defense_play(self, A, B):
        if np.random.uniform() < self.random_probability:
            self.random_defense_play(A, B)
        else:
            self.deterministic_defense_play(A, B)

    def deterministic_defense_play(self, A, B):
        potA = self.potential(A)
        potB = self.potential(B)
        if (potA >= potB):
            self.erase(A)
        else:
            self.erase(B)

    def random_defense_play(self, A, B):
        if np.random.random_sample() < 0.5:
            self.erase(A)
        else:
            self.erase(B)


    def check(self):
        """Function to chek if the game is over or not.
        Returns:
            int -- If the game is not over returns 0, otherwise returns -1 if the defender won or 1 if the attacker won.
        """

        if (sum(self.state) == 0):
            return -1
        elif (self.state[-1] >=1 ):
            return 1
        else:
            return 0

    def propose_sets(self, target):
        """Function to divide the game state into two sets A and B, where set A holds all pieces until the target level
        and set B all the pieces from the target level. The target level is then divided so that the potetials of both sets
        are as equal as possible.
        Returns:
            np.array, np.array.
        """
        if self.verbose:
            self.render()
            print("target: ",target)
        A = [0] * (self.K + 1)
        B = [0] * (self.K + 1)
        A[:target+1] = self.state[:target+1]
        B[target+1:] = self.state[target+1:]

        # amount of pieces in target level
        pieces_target = self.state[target]

        # potentials of the current splits
        potA = self.potential(A) 
        potB = self.potential(B)
        potTarget = pieces_target * self.weights[target]

        # In case that if one would add all pieces from the target level to A 
        # and A would still have a smaller potential then B, add them to A
        if potA + potTarget <= potB:
            A[target] = pieces_target

        # otherwise, if by doing so A and level target would have a greater potential then B, 
        # we ensure that the diff_pieces (the amount of pieces
        # that make the difference between A + target and B where the amount of pieces is related to the potential of the target level),
        # is a value greater then zero.
        else:
            diff_pieces = (potA + potTarget - potB)/self.weights[target]

            # if possible, we then splitt the piece difference in half and distribute them equaly.
            # If there are not enough pieces in the target level then potB + potTarget <= potA abd we put them all to B
            pieces_shift = min(int(diff_pieces/2), pieces_target)
            B[target] = pieces_shift
            A[target] = pieces_target - pieces_shift

        # because of the rounding above, to work towards more generalization, we switch the sets occasionally.
        rand = np.random.binomial(1, 0.5)
        if rand:
            return A, B
        else:
            return B, A


    def step(self, target):
        A,B = self.propose_sets(target)
        print("A: " + str(A) + ", pot: " + str(self.potential(A)))
        print("B: " + str(B) + ", pot: " + str(self.potential(B)))
        
        self.defense_play(A,B)
        win = self.check()
        if(win):
            if self.verbose:
                print("done with reward: ", win)
            self.done = 1
            self.reward = win

        return self.state, self.reward, self.done, {}


    def reset(self):
        self.state = self.random_start()
        self.done = 0
        self.reward = 0
        return self.state




    def render(self):
        for j in range(self.K + 1):
            print(self.state[j], end = " ")
        print("")


    ######################## State Space Sampling ##################################

    def random_start(self):
        return self.sample()

    def sample(self):
        """
        Samples a random start state based on initialization configuration
        """
        
        # pick sample type according to probability
        samplers = ["unif", "geo", "diverse"]
        sample_idx = np.random.multinomial(1, [self.unif_prob, self.geo_prob, self.diverse_prob])
        idx = np.argmax(sample_idx)
        sampler = samplers[idx]
        
        if sampler == "unif":
            return self.unif_sampler()
        if sampler == "geo":
            return self.geo_sampler()
        if sampler == "diverse":
            return self.diverse_sampler()        


    def get_high_one(self, state):
        """
        Takes in state and adds one piece at a high level
        """
        non_zero_idxs = [-2, -3, -4]
        idx_idxs = np.random.randint(low=0, high=3, size=10)
        for idx_idx in idx_idxs:
            non_zero_idx = non_zero_idxs[idx_idx]
            if self.potential(state) + self.weights[non_zero_idx] <= self.initial_potential:
                state[non_zero_idx] += 1
                break
        return state    
   
    def unif_sampler(self):
        """
        Samples pieces for states uniformly, for levels 0 to self.unif_high
        """
        state = np.zeros(self.K+1, dtype=int)
       
        # adds high one according to probability
        high_one = np.random.binomial(1, self.high_one_prob)
        if high_one:
            state = self.get_high_one(state)

        # checks potential of state, returning early if necessary
        if (self.initial_potential -  self.potential(state)) <= 0:
            return state
       
        # samples according to uniform probability
        pot_state = self.potential(state)

        for i in range(max(10, int(1/(100000*self.weights[0])))):
            levels = np.random.randint(low=0, high=self.unif_high, size=int(np.min([100000, 1.0/self.weights[0]])))
            # adds on each level as the potential allows
            for l in levels:
                if pot_state + self.weights[l] <= self.initial_potential:
                    state[l] += 1
                    pot_state += self.weights[l]
               
                # checks potential to break
                if pot_state >= self.initial_potential - max(1e-8, self.weights[0]):
                    break
            # checks potential to break
            if pot_state >= self.initial_potential - max(1e-8, self.weights[0]):
                break
            
        return state
            
    def geo_sampler(self):
        """
        Samples pieces for states with geometric distributions, for levels 0 to self.geo_high
        and buckets them in from lowest level to highest level
        """
        state = np.zeros(self.K+1, dtype=int)
       
        # adds high one according to probability
        high_one = np.random.binomial(1, self.high_one_prob)
        if high_one:
            state = self.get_high_one(state)
        
        # pick the p in Geometric(p), where p is randomly chosen from predefined list of ps
        ps = self.geo_ps
        p_idx = np.random.randint(low=0, high=len(ps))
        p = ps[p_idx]
        for i in range(max(1000, int(1/(100000*self.weights[0])))):
            # get pieces at different levels, highest level = self.geo_high
            assert self.K+1 < 30, "K too high, cannot use geo sampler"
            levels = np.random.geometric(p, int(1.0/self.weights[0])) - 1
            idxs = np.where(levels < self.geo_high)
            levels = levels[idxs]
            
            # bin the levels into the same place which also sorts them from 0 to K
            # counts created separately to ensure correct shape
            tmp = np.bincount(levels)
            counts = np.zeros(self.K + 1)
            counts[:len(tmp)] = tmp
            
            # add levels to state with lowest levels going first
            for l in range(self.K + 1):
                max_pieces = (self.initial_potential - self.potential(state))/self.weights[l]
                max_pieces = int(np.min([counts[l], max_pieces]))
                state[l] += max_pieces
                
                # checks potential to break
                if self.potential(state) >= self.initial_potential - max(1e-8, self.weights[0]):
                    break
            # checks potential to break
            if self.potential(state) >= self.initial_potential - max(1e-8, self.weights[0]):
                break
            
        return state
    
    def simplex_sampler(self, n):
        """ Samples n non-negative values between (0, 1) that sum to 1
        Returns in sorted order. """
        
        # edge case: n = 1
        if n == 1:
            return np.array([self.initial_potential])

        values = [np.random.uniform(low=0, high=1) for i in range(n-1)]
        values.extend([0,1])
        values.sort()
        values_arr = np.array(values)
        
        xs = values_arr[1:] - values_arr[:-1]

        # return in decresing order of magnitude, to use for higher levels
        xs = self.initial_potential*np.sort(xs)
        xs = xs[::-1]
        return xs        


    def diverse_sampler(self):
        """
        Tries to sample state to increase coverage in state space. Does this with three steps
        Step 1: Uniformly samples the number of non-zero idxs
        Step 2: Gets a set of idxs (between 0 to K-2) with size the number of nonzero idxs
                in Step 1
        Step 3: Divides up the potential available uniformly at random between the chosen idxs
        """
        
        # Sample number of nonzero idxs
        num_idxs = np.random.randint(low=1, high=self.K-1)

        # Sample actual idxs in state that are nonzero
        idxs = []
        all_states =[ i for i in  range(self.K - 1)] # can have nonzero terms up to state[K-2]
        for i in range(num_idxs):
            rand_id = np.random.randint(low=0, high=len(all_states))
            idxs.append(all_states.pop(rand_id))

        # sort idxs from largest to smallest to allocate
        # potential correctly
        idxs.sort()
        idxs.reverse()

        # allocate potential
        xs = self.simplex_sampler(num_idxs)

        # fill with appropriate number of pieces adding on any remaindr
        remainder = 0
        state = np.zeros(self.K+1, dtype=int)
        for i in range(num_idxs):
            idx = idxs[i]
            pot_idx = xs[i] + remainder
            num_pieces = int(pot_idx/self.weights[idx])
            state[idx] += num_pieces
            # update remainder
            remainder = pot_idx - num_pieces*self.weights[idx]

        return state
            
