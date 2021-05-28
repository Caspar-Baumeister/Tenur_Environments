import gym
from gym import error, spaces, utils
from gym.utils import seeding
from random import randint, choice
import numpy as np
import math


###### GYM ######

class CAttacker(gym.Env):
    """
    INPUT:
    - K: {int} amound of level
    - initial_potential: {float} initial potential
    """

    def __init__(self, K, initial_potential, error_rate=0):

        # input class variables
        self.K = K
        self.initial_potential = initial_potential
        self.error_rate = error_rate

        # action and observation space
        self.action_space = self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.K,),
            dtype=np.float32)

        self.observation_space = spaces.MultiDiscrete([10] * (K+1))

        # internal class variables
        self.weights = np.power(2.0, [-(self.K - i)
                                      for i in range(self.K + 1)])
        self.state = None
        self.done = 0
        self.reward = 0

        # start state distribution settings
        self.geo_prob = .3
        self.unif_prob = .4
        self.diverse_prob = .3
        self.high_one_prob = 0.2
        self.geo_high = self.K - 2
        self.unif_high = max(3, self.K-3)
        self.geo_ps = [0.45, 0.5, 0.6, 0.7, 0.8]

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

    def defense_play(self, A, B):
        potA = self.potential(A)
        potB = self.potential(B)

        error = np.random.uniform() < self.error_rate

        if (potA >= potB):
            if error:
                self.erase(B)
            else:
                self.erase(A)
        else:
            if error:
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
        elif (self.state[-1] >= 1):
            return 1
        else:
            return 0

    def discretize(self, target):
        A = np.zeros(self.K+1).astype("int")
        B = np.zeros(self.K+1).astype("int")
        for i, p in enumerate(target):
            A[i] = math.floor(self.state[i]*p)
            B[i] = self.state[i] - A[i]
        rand = np.random.binomial(1, 0.5)
        if rand:
            return A, B
        else:
            return B, A

    def step(self, target):
        """
        Takes the action and executes them. 
        Arguments:
            target {list} -- return from action space
        """
        A, B = self.discretize(target)

        self.defense_play(A, B)
        win = self.check()
        if(win):
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
            print(self.state[j], end=" ")
        print("")

    ######################## Start State ##################################

    def random_start(self):
        return self.sample()

    def sample(self):
        """
        Samples a random start state based on initialization configuration
        """

        # pick sample type according to probability
        samplers = ["unif", "geo", "diverse"]
        sample_idx = np.random.multinomial(
            1, [self.unif_prob, self.geo_prob, self.diverse_prob])
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
        if (self.initial_potential - self.potential(state)) <= 0:
            return state

        # samples according to uniform probability
        pot_state = self.potential(state)

        for i in range(max(10, int(1/(100000*self.weights[0])))):
            levels = np.random.randint(low=0, high=self.unif_high, size=int(
                np.min([100000, 1.0/self.weights[0]])))
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
                max_pieces = (self.initial_potential -
                              self.potential(state))/self.weights[l]
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

        values = [np.random.uniform() for i in range(n-1)]
        values.extend([0, 1])
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
        # can have nonzero terms up to state[K-2]
        all_states = [i for i in range(self.K - 1)]
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
