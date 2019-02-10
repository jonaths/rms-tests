import gym
# import gym_windy
import numpy as np
from rms.rms import RmsAlg
import sys
from plotters.plotter import PolicyPlotter
import matplotlib.pyplot as plt
from tools import tools
from random import randint
from collections import namedtuple
from tools.qlearning import LambdaRiskQLearningAgent
from tools.line_plotter import LinesPlotter
import matplotlib

args_struct = namedtuple(
    'args',
    'env_name number_steps rthres influence risk_default sim_func_name buckets')

args = args_struct(
    env_name='CartPole-v0',
    number_steps=21000,
    rthres=-1,
    influence=2,
    risk_default=0,
    sim_func_name='euclidean',
    buckets=(1, 1, 6, 12)
)

print(args)

env = gym.make(args.env_name)

actionFn = lambda state: [0, 1]
qLearnOpts = {
    'gamma': 0.9,
    'alpha': 0.1,
    'epsilon': 0.1,
    'numTraining': 1000,
    'actionFn': actionFn
}

agent = LambdaRiskQLearningAgent(**qLearnOpts)
agent.setEpsilon(0.1)

num_rows = 6
num_cols = 12
num_states = num_rows * num_cols
num_actions = 4
iteration = 0
step = 0
misc = ['rewards': [], 'steps': []]

done = False
agent.startEpisode()
s_t = env.reset()
s_t = tools.process_obs(s_t, name='buckets')

alg = RmsAlg(args.rthres, args.influence, args.risk_default, args.sim_func_name)
alg.add_to_v(s_t, tools.ind2coord(num_rows, s_t))

plotter = LinesPlotter(['reward', 'steps', 'end_state'], 1, 1000)

while True:

    if step >= args.number_steps:
        break

    if done:
        final_reward = misc['sum_reward']
        final_num_steps = len(misc['step_seq'])
        final_state = misc['step_seq'][-1]
        plotter.add_episode_to_experiment(0, iteration,
                                          [
                                              final_reward,
                                              final_num_steps,
                                              final_state
                                          ])
        agent.stopEpisode()
        agent.startEpisode()
        s_t = env.reset()
        s_t = tools.process_obs(s_t)
        iteration += 1

    # env.render()

    # action_idx = int(randint(0, 3))
    action_idx = agent.getAction(s_t)
    # action_idx = int(input('Action: '))

    obs, r, done, misc = env.step(action_idx)
    current_state = tools.process_obs(obs, name='buckets')

    alg.update(s=s_t, r=r, sprime=current_state, sprime_features=tools.ind2coord(num_rows, current_state))

    risk_penalty = abs(alg.get_risk(current_state))

    agent.observeTransition(s_t, action_idx, current_state, r, lmb=1.0, risk=risk_penalty)

    print('Output:' + ' ' + str(iteration) + ' ' + str(step) + ' ' + str(
        args.number_steps) + ' ' + str(step * 100 / args.number_steps))

    s_t = current_state
    step += 1

exp_name = 'beachworld-euclidean'
output_folder = 'output/'


tools.save_risk_map(
    alg.get_risk_dict(), num_states, env.rows, env.cols, output_folder+'riskmap-'+exp_name+'.png')
tools.save_policy(
    np.array(agent.getQTable(num_states, num_actions)), env.rows, env.cols, output_folder+'policy-'+exp_name+'.png')

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

