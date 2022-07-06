# from this import d
import numpy as np
import itertools
import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

time_format = "%H:%M:%S"
def time_now():
    return dt.datetime.now().strftime(time_format)


class PigWorld:
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
        self.__terminal_states = tuple(s for s in self.__S if self.is_terminal_state(s))
        self.__non_terminal_states = tuple(s for s in self.__S if not self.is_terminal_state(s))
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

        # inner consistency check:
        for s in all_states:
            assert self.is_valid_state(s), f'inner consistency check failed on {s}'

        print(f'built state space of length {len(all_states)}')

        return all_states

    def is_valid_state(self, s):
        valid_state_flag = True
        if s[0] > self.__winning_score - 1:
            valid_state_flag = False
        if s[1] > self.__winning_score - 1:
            valid_state_flag = False
        if s[2] > self.__max_score:
            valid_state_flag = False
        if (s[0] + s[2] > self.__max_score):
            valid_state_flag = False
        if ((self.__min_roll == 1) & (s[2] == 1)):
            valid_state_flag = False
        return valid_state_flag

    def is_terminal_state(self, s):
        is_terminal_state_flag = False
        if (s[0] + s[2] >= self.__winning_score):
            is_terminal_state_flag = True
        return is_terminal_state_flag

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
        failed_states = [sp for sp in attainable_states if not self.is_valid_state(sp)]
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
        assert self.is_valid_state(sp), assert_str

        return sp

    def get_reward(self, s):
        r = 1 if self.is_terminal_state(s) else 0
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


def value_iteration(iteration_type, env, gamma, theta):
    '''
    - implements value iteration as specified in Neller, Presser (i.e. as in Barto, Sutton)
        for the Pig Game
    - iteration_type can be one of vanilla, backward, backward_split
    - vanilla: 
        - standard value iteration that sweeps over the whole state space (or rather all the non-terminal state)
            for each run
    - backward: 
        - makes use of the fact that score_1 and score_2 can never decrease and can only change, when
            holding or throwing a 1 (e.g. s=(i, j, k) can lead to sp=(j, i, 0) but 
            s[0] + s[1] = sp[0] + sp[1])
        - computes the value function V backwards through the score_sums given by score_1 + score_2
            - computation of V for any score_sum only relies on higher score_sum, which have been
                calculated already, since higher score_sums are calculated first
        - about 50% faster on the 4-side dice 25 winning point game
        - 6-side dice 100 winning game wasnt even possible with vanilla approach

    - backward_split:
        - split up the states having the same score sum further by considering
            for i and j given all s, where ((s[0] = i and s[1] = j) or (s[0] = j and s[1] = i)) 
        - i.e. for score_sum = 90, all s with s[0] and s[1] in [45, 45] are considered and then all
            s with s[0] and s[1] in [30, 60]. Order doesnt matter here, since all the states in one 
            score_sum split can not be reached from any state with a different split and same score_sum
        - about 20% faster on the 6-side dice 100 winning point game
    '''


    print(f'{dt.datetime.now().strftime("%H:%M:%S")} Started {iteration_type} value iteration')
    # for V we use the full state space (env.S) since the q-value computation in get_q_value() 
    #   is more comprehensive if terminal state sp are available in V (even though they remain at 0)
    V = {s: 0 for s in env.S}
    # tracking the Q values is just for convenience for post-compute analysis
    Q = {s: {a: 0 for a in env.A} for s in env.non_terminal_states}

    if iteration_type == 'vanilla':
        value_iteration_core(env.non_terminal_states, env, V, Q, gamma, theta)
    
    elif iteration_type == 'backward':
        score_sum_dict = get_partition_by_score_sum(env)
        for score_sum in reversed(sorted(list(score_sum_dict.keys()))):
            print(f'{time_now()} iterate on score_sum = {score_sum} ({len(score_sum_dict[score_sum])} elements)')
            value_iteration_core(score_sum_dict[score_sum], env, V, Q, gamma, theta)

    elif iteration_type == 'backward_split':
        score_sum_dict = get_partition_by_score_sum(env)
        for score_sum in reversed(sorted(list(score_sum_dict.keys()))):
            print(f'{time_now()} iterate on score_sum = {score_sum} ({len(score_sum_dict[score_sum])} elements)')
            # score_sum = 90
            score_sum_score_switch_dict = get_score_sum_score_switch_dict(score_sum, score_sum_dict)
            for switch_states_key in score_sum_score_switch_dict.keys():
                value_iteration_core(score_sum_score_switch_dict[switch_states_key], env, V, Q, gamma, theta)

    Q = pd.DataFrame.from_dict(Q, orient='index')

    print(f'{dt.datetime.now().strftime("%H:%M:%S")} Finished {iteration_type} value iteration')
    return V, Q


def get_partition_by_score_sum(env):
    '''
    - computes all possible sum of score_1 and score_2 in env.non_terminal_states
        and returns a dict where keys are score_sums and values are a tuple of state tuples
        with score_sum equal to key
    '''
    score_sums = pd.DataFrame(env.non_terminal_states, columns=['score_1', 'score_2', 'k'])
    score_sums['score_sum'] = score_sums['score_1'] + score_sums['score_2']
    score_sums_dict = {}
    for score_sum in set(score_sums['score_sum']):
        temp = score_sums[score_sums['score_sum'] == score_sum].copy()
        temp = tuple((temp[['score_1', 'score_2', 'k']]).itertuples(index=False, name=None))
        score_sums_dict[score_sum] = temp
    return score_sums_dict


def get_score_sum_score_switch_dict(score_sum, score_sum_dict):
    '''
    - given a score_sum a dict is returned where keys are score_sum_i_j and values are all
        states with score_sum as in key and score_1/score_2 either i or j
    '''
    score_sum_states = list(score_sum_dict[score_sum])
    score_switch_dict = {}
    while score_sum_states:
        state_1_score_1 = score_sum_states[0][0]
        state_1_score_2 = score_sum_states[0][1]
        key_name = str(score_sum) + '_' + str(state_1_score_1) + '_' + str(state_1_score_2)
        score_switch_dict[key_name] = tuple(s for s in score_sum_states if (((s[0] == state_1_score_1) & (s[1] == state_1_score_2))
                                                                            | ((s[1] == state_1_score_1) & (s[0] == state_1_score_2))))
        score_sum_states = list(s for s in score_sum_states if s not in score_switch_dict[key_name])
    return score_switch_dict


def value_iteration_core(states_in_scope, env, V, Q, gamma, theta):
    '''
    - standard loop over states_in_scope until V has converged on states_in_scope
    '''
    while True:
        delta = 0
        for s in states_in_scope:
            v = V[s]
            bellman_optimality_update(env, V, Q, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return None


def bellman_optimality_update(env, V, Q, s, gamma):
    '''
    - one round of update of V at s 
    '''
    q_values_max = float("-inf")
    for a in env.A:
        q_value_a = get_q_value(env, V, s, a, gamma)
        Q[s][a] = q_value_a
        if q_value_a >= q_values_max:
            q_values_max = q_value_a    
    V[s] = q_values_max
    return None


def get_q_value(env, V, s, a, gamma):
    '''
    - computation of q-value given s and a and current state of V
    '''

    transitions = env.transitions(s, a)

    q_value = 0
    # first summand (corresponding to hold or to fail when rolling)
    sp, r, p = transitions[0]
    q_value += p * (r + gamma * (1 - V[sp]))
    # remaining summands
    for sp, r, p in transitions[1:]:
        q_value += p * (r + gamma * V[sp])
    return q_value


num_sides = 6
winning_score = 100
game_type = 'dice'
env = PigWorld(num_sides=num_sides,
               winning_score=winning_score,
               game_type=game_type)
# env.__dict__
# env.A
# env.S
# env.terminal_states
# env.non_terminal_states
gamma = 1
theta = 0.0001
V, q_values = value_iteration('backward_split', env, gamma, theta)

filename = f'q_values_{game_type}_{num_sides}_{winning_score}.gz'
q_values.to_pickle(filename)


q_values = pd.read_pickle(filename)


# score_sum_dict = get_partition_by_score_sum(env)
# for score_sum in reversed(sorted(list(score_sum_dict.keys()))):
#     print(f'{time_now()} iterate on score_sum = {score_sum} ({len(score_sum_dict[score_sum])} elements)')
# score_sum_dict[195]

# s = (5,7,3)
# q_values.loc[s,:]


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

# derive switches between actions
# first sort (should be sorted but just to be safe)
decision_space.sort_values(by=['score_1', 'score_2', 'turn_total'], inplace=True)
# shift and compare with previous
decision_space['next_step'] = decision_space.groupby(['score_1', 'score_2'])['roll_flag'].shift(-1)
# label boundaries
flag_temp = decision_space['roll_flag'] != decision_space['next_step']
flag_temp = flag_temp & (~np.isnan(decision_space['next_step']))
decision_space['boundary_flag'] = 0
decision_space.loc[flag_temp, 'boundary_flag'] = 1
decision_space
del decision_space['next_step']

# look at some decision boundaries for score_1 and score_2 fixed
score_1 = 70
score_2 = 30
temp_flag = ((decision_space['score_1'] == score_1) & (decision_space['score_2'] == score_2))
decision_line = decision_space[temp_flag].copy()
decision_line

# look at some decision boundaries for score_2 fixed
score_2 = 30
temp_flag = decision_space['score_2'] == score_2
decision_line = decision_space[temp_flag].copy()
decision_line
# endregion


# region plot decision boundary for score_2 fixed
score_2 = 30
temp_flag = ((decision_space['score_2'] == score_2) & (decision_space['boundary_flag'] == 1))
decision_line = decision_space[temp_flag].copy()
decision_line
plt.scatter(x=decision_line['score_1'], y=decision_line['turn_total'])
plt.plot(list(decision_line['score_1']), list(decision_line['turn_total']))
# endregion


# region plot decision boundary with plotly
temp_flag = decision_space['boundary_flag'] == 1
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
temp_flag = decision_space['boundary_flag'] == 1
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
