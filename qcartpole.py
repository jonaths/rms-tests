import gym
import numpy as np
import math
from collections import deque
import sys

from rms.rms import RmsAlg

from tools import tools
from tools.history import History
from tools.line_plotter import LinesPlotter

import matplotlib.pyplot as plt
import matplotlib


class QCartPoleSolver():
    def __init__(self, buckets=(1, 1, 6, 12,), n_episodes=1000, n_win_ticks=195, min_alpha=0.1,
                 min_epsilon=0.1, gamma=1.0, ada_divisor=25, max_env_steps=None, quiet=False,
                 monitor=False):
        self.buckets = buckets  # down-scaling feature space to discrete range
        self.n_episodes = n_episodes  # training episodes
        self.n_win_ticks = n_win_ticks  # average ticks over 100 episodes required for win
        self.min_alpha = min_alpha  # learning rate
        self.min_epsilon = min_epsilon  # exploration rate
        self.gamma = gamma  # discount factor
        self.ada_divisor = ada_divisor  # only for development purposes
        self.quiet = quiet
        self.plotter = LinesPlotter(['reward', 'steps', 'end_state'], 1, n_episodes)
        self.history = History()

        self.alg = RmsAlg(-1, 2, 0, 'euclidean')
        self.danger_states = []
        for i in range(6 * 12):
            if tools.is_top_or_bottom(i):
                self.danger_states.append(i)

        self.danger_states.append(1)
        self.danger_states.append(70)


        self.env = gym.make('CartPole-v0')
        if max_env_steps is not None:
            self.env._max_episode_steps = max_env_steps
        if monitor:
            self.env = gym.wrappers.Monitor(self.env, 'tmp/cartpole-1',
                                            force=True)  # record results for upload

        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2],
                        math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2],
                        -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in
                  range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(
            self.Q[state])

    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (
                    reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def get_epsilon(self, t):
        # return 0.1
        comp = math.log10((t + 1) / float(self.ada_divisor * 1.))
        return max(self.min_epsilon, min(1.0, 1.0 - comp))

    def get_alpha(self, t):
        # return 0.9
        comp = math.log10((t + 1) / float(self.ada_divisor * 1.))
        return max(self.min_alpha, min(1.0, 1.0 - comp))

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            current_state = self.discretize(self.env.reset())
            self.alg.add_to_v(
                tools.coord2ind(current_state[-2:], 6, 12),
                current_state)

            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)
            done = False
            i = 0

            while not done:
                # self.env.render()
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize(obs)
                self.history.insert((current_state, action, reward, new_state))

                current_state_ind = tools.coord2ind(current_state[-2:], 6, 12)
                s_prime_ind = tools.coord2ind(new_state[-2:], 6, 12)
                self.alg.update(
                    s=current_state_ind,
                    r=-10 if s_prime_ind in self.danger_states else reward,
                    sprime=s_prime_ind,
                    sprime_features=new_state)
                risk_penalty = abs(self.alg.get_risk(current_state_ind))
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                i += 1

            last_state_as_ind = tools.coord2ind(self.history.get_state_sequence()[-1][-2:], 6, 12)
            self.plotter.add_episode_to_experiment(0, e,
                                                   [
                                                       self.history.get_total_reward(),
                                                       self.history.get_steps_count(),
                                                       last_state_as_ind

                                                   ])
            self.history.clear()
            # scores.append(i)
            # mean_score = np.mean(scores)
            # if mean_score >= self.n_win_ticks and e >= 100:
            #     if not self.quiet:
            #         print('Ran {} episodes. Solved after {} trials'.format(e, e - 100))
            #     return e - 100
            # if e % 100 == 0 and not self.quiet:
            #     print(
            #         '[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(
            #             e, mean_score))

        # if not self.quiet:
        #     print('Did not solve after {} episodes'.format(e))
        return e

    def plot(self):

        output_folder = 'output/'

        self.plotter.save_data(output_folder+'data')


        print(self.danger_states)

        exp_name = 'cartpole-euclidean'

        tools.save_risk_map(
            self.alg.get_risk_dict(), 6*12, 6, 12,
            output_folder + 'riskmap-' + exp_name + '.png')


        fig, ax = self.plotter.get_var_line_plot(['reward', 'steps'], 'average', window_size=50)
        fig.legend()
        plt.tight_layout()
        plt.savefig(output_folder + 'reward-steps.png')

        # aqui voy... no grafica el historial de fallos

        fig, ax = self.plotter.get_var_cummulative_matching_plot('end_state', self.danger_states)
        fig.legend()
        plt.tight_layout()
        plt.savefig(output_folder + 'end-cummulative-' + exp_name + '.png')


if __name__ == "__main__":
    solver = QCartPoleSolver()
    solver.run()
    solver.plot()
