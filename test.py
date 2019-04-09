import matplotlib.pyplot as plt
import numpy as np

from tools.line_plotter import LinesPlotter

plotter = LinesPlotter(['reward', 'steps', 'end_state'], 2, 10)
iteration = 0
plotter.add_episode_to_experiment(0, iteration,
                                  [
                                      1,
                                      2,
                                      3
                                  ])
iteration = 1
plotter.add_episode_to_experiment(0, iteration,
                                  [
                                      4,
                                      5,
                                      6
                                  ])
iteration = 2
plotter.add_episode_to_experiment(0, iteration,
                                  [
                                      5,
                                      5,
                                      5
                                  ])


print(plotter.data)
print(plotter.data.shape)
summary = np.average(plotter.calculate_summary('average').get_var_from_summary('steps')[0, iteration - 2: iteration])
print(summary)
print(summary.shape)
