import numpy as np
from plotters.plotter import PolicyPlotter
import matplotlib.pyplot as plt


def create_states_list_from_dict(states_dict, num_states):
    states_list = []
    for i in range(num_states):
        states_list.append(0)
        if i in states_dict.keys():
            states_list[i] = states_dict[i]
    return states_list


def process_obs(obs, params=None, name=None):
    if name == 'grid':
        return obs
    else:
        return obs


def save_risk_map(states_dict, num_states, num_rows, num_cols, name):
    # states_dict = {0: 5, 1: 2}
    states_list = np.array(create_states_list_from_dict(states_dict, num_states)).reshape(
        (num_states, 1, -1))

    plotter = PolicyPlotter(states_list, num_rows, num_cols)

    # tambien se puede generar el mapa de politicas
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    im, cbar = plotter.build_heatmap(annotate=False, cmap='Reds')
    fig.tight_layout()
    fig.savefig(name, dpi=100)


def save_policy(qtable, num_rows, num_cols, name):

    plotter = PolicyPlotter(qtable, num_rows, num_cols)

    # tambien se puede generar el mapa de politicas
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    im, cbar, text = plotter.build_policy(
        labels=['^', '>', 'v', '<'], show_numbers=False, cmap='Blues')
    fig.tight_layout()
    fig.savefig(name, dpi=100)