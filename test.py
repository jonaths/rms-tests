
from tools.line_plotter import LinesPlotter
import random
import numpy as np
import matplotlib.pyplot as plt


num_experiments = 2
num_episodes = 5
plotter = LinesPlotter(['r', 's', 'f'], num_experiments, num_episodes)


for experiment in range(num_experiments):
    for episode in range(num_episodes):
        plotter.add_episode_to_experiment(experiment, episode, np.array([1, random.randint(0, 5), random.randint(0, 3)]))

print('XXX')
summary = plotter.calculate_summary()
print(summary)

print('XXX')
summary = plotter.calculate_summary(func='max').get_var_from_summary(var_name='r')
print(summary)

fig, ax = plotter.get_var_line_plot(['r', 's', 'f'], 'average')
fig.legend()
plt.tight_layout()
plt.show()