import numpy as np
import sys
import matplotlib.pyplot as plt


class LinesPlotter:

    def __init__(self, var_names_list, num_experiments, num_episodes):
        pass
        self.var_names_list = var_names_list
        self.num_episodes = num_episodes
        self.num_experiments = num_experiments
        self.data = np.zeros(shape=(num_experiments, num_episodes, len(var_names_list)))
        self.summary = None

    def add_episode_to_experiment(self, experiment, episode, var_values):
        """
        Agrega un episodio al experimento
        :param experiment: el numero de experimento
        :param episode: el numero de episodio
        :param var_values: una lista de valores en el mismo orden que var_name_list
        :return:
        """
        self.data[experiment, episode, :] = np.array(var_values)
        return self

    def calculate_summary(self, func='average'):
        """
        Crea un summary de los valores almacenados en el historial
        :param func: la operacion de summary (average, max)
        :return:
        """
        temp = np.transpose(self.data, (2, 1, 0))
        if func is 'average':
            self.summary = np.average(temp, axis=2)
        elif func is 'max':
            self.summary = np.max(temp, axis=2)
        else:
            raise Exception('Invalid summary operation')

        return self

    def get_var_from_summary(self, var_name):
        # Si se pasa un valor regresa unicamente ese elemento
        if var_name in self.var_names_list:
            index = self.var_names_list.index(var_name)
            summary = np.expand_dims(self.summary[index], axis=0)
        return summary

    def get_var_line_plot(self, var_name_list, func, window_size=10):
        fig, ax = plt.subplots()
        self.calculate_summary(func)
        for var in var_name_list:
            data = self.get_var_from_summary(var)[0]

            data = np.pad(data, (window_size // 2, window_size - 1 - window_size // 2), mode='edge')
            data = np.convolve(data, np.ones((window_size,)) / window_size, mode='valid')

            ax.plot(range(self.num_episodes), data, label=var)

        return fig, ax
