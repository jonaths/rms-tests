
from tools.line_plotter import LinesPlotter
import random
import numpy as np
import matplotlib.pyplot as plt


num_experiments = 2
num_episodes = 5
plotter = LinesPlotter(['r', 's', 'f'], num_experiments, num_episodes)


for experiment in range(num_experiments):
    for episode in range(num_episodes):
        plotter.add_episode_to_experiment(experiment, episode, np.array([1, random.randint(0, 5), random.randint(0, 5)]))

# plotter = LinesPlotter.load_data('data.npy', ['r', 's', 'f'])

print(plotter.data)

summary = plotter.calculate_summary()
print(summary)

summary = plotter.calculate_summary(func='average').get_var_from_summary(var_name='f')
print(summary)

# fig, ax = plotter.get_var_line_plot(['r', 's'], 'average')
# fig.legend()
# plt.tight_layout()
# plt.show()

print('XXX')
# fig, ax = plotter.get_pie_plot('f', mapping_dict={'safe': [0, 1], 'unsafe': [2]})
# fig.legend()
# plt.tight_layout()
# plt.show()

fig, ax = plotter.get_var_cummulative_matching_plot('f', [1, 2])
fig.legend()
plt.tight_layout()
plt.show()

# plotter.save_data('data')
