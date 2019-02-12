import gym
import gym_windy
import numpy as np
from rms.rms import RmsAlg
import sys
from plotters.plotter import PolicyPlotter
import matplotlib.pyplot as plt
from tools import tools
from tools.history import History
from random import randint
from collections import namedtuple
from tools.qlearning import LambdaRiskQLearningAgent
from tools.line_plotter import LinesPlotter
import matplotlib

args_struct = namedtuple(
    'args',
    'env_name number_steps rthres influence risk_default sim_func_name')

args = args_struct(
    env_name='border-v0',
    number_steps=21000,
    rthres=-1,
    influence=2,
    risk_default=0,
    sim_func_name='euclidean'
)

print(args)

env = gym.make(args.env_name)

actionFn = lambda state: env.get_possible_actions(state)
qLearnOpts = {
    'gamma': 0.9,
    'alpha': 0.1,
    'epsilon': 0.1,
    'numTraining': 1000,
    'actionFn': actionFn
}

agent = LambdaRiskQLearningAgent(**qLearnOpts)
agent.setEpsilon(0.1)

num_states = env.cols * env.rows
num_actions = 4
iteration = 0
step = 0

done = False
agent.startEpisode()
s_t = env.reset()
s_t = tools.process_obs(s_t, name='grid')

alg = RmsAlg(args.rthres, args.influence, args.risk_default, args.sim_func_name)
alg.add_to_v(s_t, env.ind2coord(s_t))

plotter = LinesPlotter(['reward', 'steps', 'end_state'], 1, 1000)
history = History()

while True:

    if step >= args.number_steps:
        break

    if done:

        plotter.add_episode_to_experiment(0, iteration,
                                          [
                                              history.get_total_reward(),
                                              history.get_steps_count(),
                                              history.get_state_sequence()[-1]
                                          ])
        history.clear()
        agent.stopEpisode()
        agent.startEpisode()
        s_t = env.reset()
        s_t = tools.process_obs(s_t, name='grid')
        iteration += 1

    # env.render()

    # action_idx = int(randint(0, 3))
    action_idx = agent.getAction(s_t)
    # action_idx = int(input('Action: '))

    obs, r, done, misc = env.step(action_idx)
    obs = tools.process_obs(obs, name='grid')
    history.insert((s_t, action_idx, r, obs))

    alg.update(s=s_t, r=r, sprime=obs, sprime_features=env.ind2coord(obs))

    risk_penalty = abs(alg.get_risk(obs))

    agent.observeTransition(s_t, action_idx, obs, r - risk_penalty, lmb=1.0, risk=risk_penalty)

    print('Output:' + ' ' + str(iteration) + ' ' + str(step) + ' ' + str(
        args.number_steps) + ' ' + str(step * 100 / args.number_steps))

    s_t = obs
    step += 1

exp_name = 'beachworld-euclidean'
output_folder = 'output/'



tools.save_risk_map(
    alg.get_risk_dict(), num_states, env.rows, env.cols, output_folder+'riskmap-'+exp_name+'.png')
tools.save_policy(
    np.array(agent.getQTable(num_states, num_actions)),
    env.rows,
    env.cols, output_folder+'policy-'+exp_name+'.png',
    labels=['^', '>', 'v', '<'])

plotter.save_data(output_folder+'data')

matplotlib.rcParams.update({'font.size': 22})

fig, ax = plotter.get_var_line_plot(['reward', 'steps'], 'average', window_size=50)
fig.legend()
plt.tight_layout()
plt.savefig(output_folder+'reward-steps.png')

fig, ax = plotter.get_pie_plot('end_state',
                               mapping_dict={
                                   'safe': [env.finish_state_one],
                                   'unsafe': env.hole_state})
plt.tight_layout()
plt.savefig(output_folder+'end-reasons-'+exp_name+'.png')

fig, ax = plotter.get_var_cummulative_matching_plot('end_state', env.hole_state)
fig.legend()
plt.tight_layout()
plt.savefig(output_folder+'end-cummulative-'+exp_name+'.png')

