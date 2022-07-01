import numpy as np
import itertools
import pandas as pd
import datetime as dt



class PigWorldLooped:
    def __init__(self,
                 num_sides=2,
                 winning_score=2,
                 game_type='coin'
                 ):

        if game_type not in ['coin', 'dice']:
            raise ValueError("game_type takes only 'coin' or 'dice'")
        if not (((game_type == 'coin') & (num_sides == 2)) | (game_type == 'dice')):
            raise ValueError('wrong combinaton of game_type and num_side')

        self.__num_sides = num_sides
        self.__winning_score = winning_score
        self.__game_type = game_type

        self.__min_roll = 0 if self.__game_type == 'coin' else 1
        self.__max_roll = num_sides if self.__min_roll == 1 else self.__num_sides - 1
        self.__max_score = self.__winning_score - 1 + self.__max_roll

        self.__roll_values = list(range(self.__min_roll, self.__max_roll + 1))
        self.__roll_values[0] = 0  # has an effect only when game_type == 'dice'
        self.__roll_values = tuple(self.__roll_values)

        self.__A = ('roll', 'hold')
        print('building state space')
        self.__S = self._build_state_space(self.__winning_score,
                                           self.__max_score,
                                           self.__min_roll)
        print('get terminal states')
        self.__terminal_states = tuple(s for s in self.__S if s[0] + s[2] >= self.__winning_score)
        print('get none terminal states')
        self.__non_terminal_states = tuple(s for s in self.__S if s not in self.__terminal_states)
        
        print('finished building env')

    def _build_state_space(self,
                           winning_score,
                           max_score,
                           min_roll):
        # initial definition of state tuples (i, j, k)
        # since winning states are a combination of current score i >= 0 
        # AND turn value k > 0, we only need i and j to go up to winning_score - 1, which 
        # is taken care of by the range definition
        all_states = tuple(itertools.product(range(0, winning_score, 1),
                                            range(0, winning_score, 1),
                                            range(0, max_score + 1, 1)))

        # remove all states where the agents current score plus 
        # his current turn value are higher than the max score
        all_states = tuple(s for s in all_states if s[0] + s[2] <= max_score)

        # if min_roll == 1 , i.e. game_type == 'dice', then turn value k
        # can never equal 1, since 1 deletes turn value and 2 is lowest roll value
        # to add to k
        if min_roll == 1:
            all_states = tuple(s for s in all_states if not (s[2] == 1))

        print(f'built state space of length {len(all_states)}')

        return all_states

    def get_attainable_states_after_roll(self, s):
        ''''
        attainable values from (i, j, k) are:
        (j, i, 0), (i, j, k+1), (i, j, k+2), ... , (i, j, k+max_roll)
        s = (0, 1, 1)    
        '''
        # create the first attainable state, which is the one resulting
        # from rolling a 1 / throwing a tail, i.e. from a fail
        sp_fail = (s[1], s[0], 0)

        # create the attainble states beyond the failed roll/throw
        sp_success = tuple((s[0], s[1], s[2] + r) for r in self.__roll_values[1:])

        # append fail and success states
        attainable_states = (sp_fail, ) + sp_success

        # check that attainable states are within the state space!
        failed_states = [sp for sp in attainable_states if sp not in self.__S]
        assert_str = f'input {s} created attainable state(s) {failed_states} not in state space'
        assert not failed_states, assert_str

        return attainable_states

    def get_attainable_state_after_hold(self, s):
        '''
        hold turns (i, j, k) into (j, i + k, 0)
        '''

        sp = (s[1], s[0] + s[2], 0)
        
        # check that attainable state is within the state space!
        assert_str = f'input {s} created attainable state {sp} not in state space'
        assert sp in self.__S, assert_str

        return sp

    def get_reward(self, s):
        r = 1 if s in self.__terminal_states else 0
        return r

    def transitions(self, s, a):
        '''
        - sps stand for the set of s' (s primes) reachable from state s
        - computes p(s', r | s, a) given s and a
        ''' 
        if a == 'roll':
            sps = self.get_attainable_states_after_roll(s)
            # define p(s', r | s, a) (fair coin/dice is assumed)
            probs = tuple(1/self.__num_sides for r in self.__roll_values)
            rewards = tuple(self.get_reward(sp) for sp in sps)
            # combining into one tuple of tuples for better indexing later
            transitions = tuple(zip(sps, rewards, probs))

        elif a == 'hold':
            sp = self.get_attainable_state_after_hold(s)
            prob = 1
            reward = self.get_reward(sp)
            # combining into one tuple of tuples for better indexing later
            transitions = ((sp, reward, prob),)

        return transitions

    @property
    def A(self):
        return self.__A

    @property
    def S(self):
        return self.__S

    @property
    def terminal_states(self):
        return self.__terminal_states

    @property
    def non_terminal_states(self):
        return self.__non_terminal_states


def value_iteration(env, gamma, theta):
    print(f'{dt.datetime.now().strftime("%H:%M:%S")} Stated value iteration')

    V = {s: 0 for s in env.S}
    log_counter = 0
    while True:
        log_counter += 1
        delta = 0
        for s in env.non_terminal_states:
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
        
        print(f'{dt.datetime.now().strftime("%H:%M:%S")} Loop {log_counter} finshed. Delta at {delta}') if log_counter%5 == 0 else None

    q_values = get_q_values(env, V, gamma)

    return V, q_values


def bellman_optimality_update(env, V, s, gamma):
    q_values_max = float("-inf")
    for a in env.A:
        q_value_a = get_q_value(env, V, s, a, gamma)
        if q_value_a >= q_values_max:
            q_values_max = q_value_a    
    V[s] = q_values_max


def get_q_value(env, V, s, a, gamma):

    transitions = env.transitions(s, a)

    q_value = 0
    # first summand (corresponding to hold or to fail when rolling)
    sp, r, p = transitions[0]
    q_value += p * (r + gamma * (1 - V[sp]))
    # remaining summands
    for sp, r, p in transitions[1:]:
        q_value += p * (r + gamma * V[sp])

    return q_value


def get_q_values(env, V, gamma):
    q_values = pd.DataFrame(columns=env.A, index=env.non_terminal_states)
    for s in env.non_terminal_states:
        for a in env.A:
            q_value_a = get_q_value(env, V, s, a, gamma)
            q_values.loc[s, a] = q_value_a
    return q_values


env = PigWorldLooped(num_sides=4,
                     winning_score=25,
                     game_type='dice')
# env = PigWorld(winning_score=10)
# env.__dict__
# env.A
# env.S
# env.terminal_states
# env.non_terminal_states
gamma = 1
theta = 0.001
V, q_values = value_iteration(env, gamma, theta)

s = (5,7,3)
q_values.loc[s,:]

print(q_values)
# V

# q_values.head(10)










