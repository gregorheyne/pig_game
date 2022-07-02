# from this import d
import numpy as np
import itertools
import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go

time_format = "%H:%M:%S"
def time_now():
    return dt.datetime.now().strftime(time_format)

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
        print(f'{time_now()} building state space')
        self.__S = self._build_state_space(self.__winning_score,
                                           self.__max_score,
                                           self.__min_roll)
        self.__terminal_states = tuple(s for s in self.__S if s[0] + s[2] >= self.__winning_score)
        self.__non_terminal_states = tuple(s for s in self.__S if s[0] + s[2] < self.__winning_score)
        print(f'{time_now()} finished building env')

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


def value_iteration_vanilla(env, gamma, theta):
    print(f'{dt.datetime.now().strftime("%H:%M:%S")} Started vanilla value iteration')
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
        
        print(f'{time_now()} Loop {log_counter} finished. Delta at {delta}') if log_counter%5 == 0 else None

    q_values = get_q_values(env, V, gamma)

    print(f'{dt.datetime.now().strftime("%H:%M:%S")} Finished vanilla value iteration')
    return V, q_values


def value_iteration_backward(env, gamma, theta):
    print(f'{dt.datetime.now().strftime("%H:%M:%S")} Started backward value iteration')

    V = {s: 0 for s in env.S}

    score_sum_dict = get_partition_by_score_sum()
    for score_sum in reversed(sorted(list(score_sum_dict.keys()))):
        print(f'{time_now()} iterate on score_sum = {score_sum} ({len(score_sum_dict[score_sum])} elements)')
        while True:
            delta = 0
            for s in score_sum_dict[score_sum]:
                v = V[s]
                bellman_optimality_update(env, V, s, gamma)
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

    q_values = get_q_values(env, V, gamma)

    print(f'{dt.datetime.now().strftime("%H:%M:%S")} Finished backward value iteration')
    return V, q_values


def get_partition_by_score_sum():
    score_sums = pd.DataFrame(env.non_terminal_states, columns=['score_1', 'score_2', 'k'])
    score_sums['score_sum'] = score_sums['score_1'] + score_sums['score_2']
    score_sums_dict = {}
    for score_sum in set(score_sums['score_sum']):
        temp = score_sums[score_sums['score_sum'] == score_sum].copy()
        temp = tuple((temp[['score_1', 'score_2', 'k']]).itertuples(index=False, name=None))
        score_sums_dict[score_sum] = temp
    return score_sums_dict


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


# env = PigWorldLooped(num_sides=4,
#                      winning_score=25,
#                      game_type='dice')
env = PigWorldLooped(winning_score=10)
# env.__dict__
# env.A
# env.S
# env.terminal_states
# env.non_terminal_states
gamma = 1
theta = 0.0001
# V, q_values = value_iteration_vanilla(env, gamma, theta)
V, q_values = value_iteration_backward(env, gamma, theta)

s = (5,7,3)
q_values.loc[s,:]


# region get decision_space df
# generate roll_flag
decision_space = q_values.copy()
roll_flag = decision_space['roll'] >= decision_space['hold']
decision_space['roll_flag'] = roll_flag
decision_space['roll_flag'] = decision_space['roll_flag'].astype('int')

# turn index into columns
cols_temp = list(decision_space.columns)
decision_space.reset_index(inplace=True, drop=False)
decision_space.columns = ['score_1', 'score_2', 'turn_total'] + cols_temp
# derive highest k where still rolling given score_1 and score_2
decision_space['max_turn_total'] = decision_space.groupby(by=['score_1', 'score_2', 'roll_flag'])['turn_total'].transform(max)

# look at some decision boundaries for score_1 and score_2 fixed
score_1 = 1
score_2 = 8
temp_flag = ((decision_space['score_1'] == score_1) & (decision_space['score_2'] == score_2))
decision_line = decision_space[temp_flag].copy()
decision_line
# endregion


# region plot decision boundary with plotly
temp_flag = ((decision_space['roll_flag'] == 1)
             & ((decision_space['turn_total'] == decision_space['max_turn_total'])))
decision_boundary = decision_space[temp_flag].copy()

# transform data into shape of data used in the plotly example on
# https://plotly.com/python/3d-surface-plots/
# axis annotation taken from https://plotly.com/python/3d-axes/
decision_boundary.set_index(keys=['score_1', 'score_2'], inplace=True)
decision_boundary = decision_boundary[['turn_total']]
decision_boundary = decision_boundary.unstack()
decision_boundary = decision_boundary.T
print(decision_boundary)

fig = go.Figure(data=[go.Surface(z=decision_boundary.values)])
fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
fig.update_layout(title='Decision boundary', 
                    scene = dict(
                    xaxis_title='x = Score_1',
                    yaxis_title='y = Score_2',
                    zaxis_title='z = Turn Value k'),
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10))
fig.show()
# endregion


# region matplotlib plot as in
# https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
# using Emanuels solution
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# prepare data
temp_flag = ((decision_space['roll_flag'] == 1)
             & ((decision_space['turn_total'] == decision_space['max_turn_total'])))
decision_boundary = decision_space[temp_flag].copy()
x = decision_boundary['score_1']
y = decision_boundary['score_2']
z = decision_boundary['turn_total']

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.savefig('teste.pdf')
plt.show()
# endregion


# region surface plot with matplotlib
# https://matplotlib.org/stable/gallery/mplot3d/surface3d.html
# to make this example work in our context look at the solution
# of Steven in
# https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator

# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
# # Plot the surface.
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()
# endregion
