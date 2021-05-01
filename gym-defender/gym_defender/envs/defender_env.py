import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from random import randint, choice
import numpy as np

class Defender(gym.Env):

    def __init__(self, K, initial_potential, disjoint_support_probabillity=0.5, verbose = 0, 
    geo_prob = .3, unif_prob = .4, diverse_prob = .3, high_one_prob = 0.2, ):

        self.K = K
        self.initial_potential = initial_potential
        
        self.disjoint_support_probabillity = disjoint_support_probabillity
        self.verbose = verbose

        # internel variables
        self.weights = np.power(2.0, [-(self.K - i) for i in range(self.K + 1)])
        self.done = 0
        self.reward = 0
        self.state = None
        self.game_state = None

        # spaces
        self.action_space = spaces.Discrete(2)
        # multidiscrete action space upper bound should be:
        # 1.2 = the potential multiplicator (if potential is greater, there can be more pieces in each level.)
        # [int(1.2 * (2**(min(15,K)-i))) for i in range(min(15,K)+1)] * 2 
        self.observation_space= spaces.MultiDiscrete([10]* (2*K+2))

        # random initial gamestate settings
        self.geo_prob = geo_prob
        self.unif_prob = unif_prob
        self.diverse_prob = diverse_prob
        self.high_one_prob = high_one_prob
        self.geo_high = self.K - 2
        self.unif_high = max(3, self.K-3)
        self.geo_ps = [0.45, 0.5, 0.6, 0.7, 0.8]
        

    def potential(self, A):
        """Function to calculates the potential of a set A 
        Arguments:
            A {list} -- The state, from that we want to know the potential
        """
        return np.sum(A*self.weights)


    def split(self, A):
        B = [z - a for z, a in zip(self.game_state, A)]
        return A, B


    def erase(self, A):
        """Function to remove the partition A from the game state
        Arguments:
            A {list} -- The list representing the partition we want to remove
        """

        self.game_state = [z - a for z, a in zip(self.game_state, A)]
        self.game_state = [0] + self.game_state[:-1] 


    def disjoint_support(self):
        A = np.zeros(self.K+1, dtype=int)
        B = np.zeros(self.K+1, dtype=int)

        nonzeros = np.where(np.array(self.game_state) > 0)[0]
        thresholds = [1./3, 5./16, 14./32]
        _ = np.random.multinomial(3, [0.8, 0.1, 0.1])
        _ = np.argmax(_)
        threshold = thresholds[_]
        idxs = nonzeros[np.random.permutation(len(nonzeros))]

        potA = self.potential(A)
        potB = self.potential(B)
        for idx in idxs:
            l_pieces = self.game_state[idx]
            # check to see what potential of pieces is
            # if potential very large, fraction, equally divide
            if l_pieces*self.weights[idx] >= self.initial_potential/2.:
                # try to equally divide
                if l_pieces % 2 == 0:
                    pieces = int(l_pieces/2)
                    A[idx] += pieces
                    B[idx] += pieces
                    potA += (l_pieces*self.weights[idx])/2.
                    potB += (l_pieces*self.weights[idx])/2.
                else:
                    A[idx] += int(l_pieces/2)
                    B[idx] += (int(l_pieces/2) + 1)
                    potA += int(l_pieces/2)*self.weights[idx]
                    potB += (int(l_pieces/2) + 1)*self.weights[idx]

            else:
                if potA >= threshold*self.initial_potential:
                    B[idx] += l_pieces
                    potB += l_pieces*self.weights[idx]
                else:
                    A[idx] += l_pieces
                    potA += l_pieces*self.weights[idx]
        # vary which of A or B is underweighted set
        p = np.random.uniform(low=0, high=1)
        if p >= 0.5:
            return B, A
        else:
            return A, B
            
    def optimal_split(self):

        A = np.zeros(self.K+1, dtype=int)
        B = np.zeros(self.K+1, dtype=int)

        levels = [i for i in range(self.K + 1)]
        levels.reverse()

        for l in levels:
            l_pieces = self.game_state[l]
            if l_pieces == 0:
                continue

            weight = self.weights[l]
            l_weight = l_pieces*weight
            potA = self.potential(A)
            potB = self.potential(B)

            # divide equally at that level if potentials are equal
            if potA == potB:
                A, B = self.equal_divide(A, B, potA, potB, l, l_pieces)

             # if potentials are not equal
            else:
                diff = np.abs(potA - potB)
                num_pieces = np.ceil(diff/weight).astype("int")

                # if the number of pieces which are the difference is less than l_pieces
                if num_pieces <= l_pieces:
                    diff_pieces = num_pieces

                    if potA < potB:
                        A[l] += diff_pieces
                    else:
                        B[l] += diff_pieces

                    l_pieces -= diff_pieces
                    A, B = self.equal_divide(A, B, potA, potB, l, l_pieces)
                else:
                    if potA < potB:
                        A[l] += l_pieces
                    else:
                        B[l] += l_pieces
        return A, B

    def equal_divide(self, A, B, potA, potB, l, l_pieces):
        # divides up pieces when potA, potB are equal except off by 1

        if l_pieces % 2 == 0:
            A[l] += l_pieces/2
            B[l] += l_pieces/2

        else:
            larger = np.ceil(l_pieces/2)
            smaller = np.floor(l_pieces/2)
            assert larger + smaller == l_pieces, print("division incorrect",
                                                       larger, smaller, l_pieces)

            if potA < potB:
                A[l] += larger
                B[l] += smaller

            elif potB < potA:
                A[l] += smaller
                B[l] += larger

            else:
                prob_A = np.random.binomial(1, 0.5)
                if prob_A:
                    A[l] += larger
                    B[l] += smaller
                else:
                    A[l] += smaller
                    B[l] += larger

        return A, B

    def attacker_play(self):
        # if only few pieces left, play optimally
        num_idxs = np.sum(self.game_state)
        if num_idxs <= 3:
            return self.optimal_split()
        # otherwise play according to difficulty
        if randint(1,100)<=self.disjoint_support_probabillity:
            return self.disjoint_support()
        else:
            return self.optimal_split()


    def check(self):
        """Function to chek if the game is over or not.
        Returns:
            int -- If the game is not over returns 0, otherwise returns 1 if the defender won or -1 if the attacker won.
        """

        if (sum(self.game_state) == 0):
            return 1
        elif (self.game_state[-1] >=1 ):
            return -1
        else:
            return 0


    def step(self, target):
        
        A = self.state[: self.K + 1]
        B = self.state[self.K + 1 :]
        if (target == 0):
            self.erase(A)
        else:
            self.erase(B)
        win = self.check()
        if (win):
            self.done = 1
            self.reward = win

        if self.done != 1:
            A, B = self.attacker_play()
            if self.verbose >= 2:
                print("erase: ", target)
                print("new potentials: A: ", self.potential(A), ", B: " , self.potential(B))
            self.state = np.concatenate([A,B])

        return self.state, self.reward, self.done, {}


    def reset(self):
        self.game_state = self.random_start()
        self.done = 0
        self.reward = 0
        A, B = self.attacker_play()
        if self.verbose >= 2:
            print("")
            print("RESET")
            print("start potentials: A: ", self.potential(A), ", B: " , self.potential(B))
        self.state = np.concatenate([A,B])
        return self.state

    def render(self):
        for j in range(self.K + 1):
            print(self.game_state[j], end = " ")
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

        values = [np.random.uniform() for i in range(n-1)]
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
