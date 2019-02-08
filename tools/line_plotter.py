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
        if experiment >= self.num_experiments or episode >= self.num_episodes:
            return self
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
            summary_pickled = np.expand_dims(self.summary[index], axis=0)
        else:
            raise Exception('Invalid var_name. ')
        return summary_pickled

    def get_var_from_data(self, var_name):
        # Si se pasa un valor regresa unicamente ese elemento
        if var_name in self.var_names_list:
            index = self.var_names_list.index(var_name)
            data_pickled = self.data[:, :, index]
        else:
            raise Exception('Invalid var_name. ')
        return data_pickled

    def get_var_line_plot(self, var_name_list, func, window_size=20):
        fig, ax = plt.subplots()
        self.calculate_summary(func)
        for var in var_name_list:
            data = self.get_var_from_summary(var)[0]

            data = np.pad(data, (window_size // 2, window_size - 1 - window_size // 2), mode='edge')
            data = np.convolve(data, np.ones((window_size,)) / window_size, mode='valid')

            ax.plot(range(self.num_episodes), data, label=var)

        return fig, ax

    def get_var_cummulative_matching_plot(self, var_name, matching_list):
        """
        Recibe matching_list y verifica los valores en var_name de data. Genera un acumulado.
        :param var_name: el nombre de la variable a contar
        :param matching_list: los valores que cuentan como 1
        :return:
        """

        fig, ax = plt.subplots()
        data = self.get_var_from_data(var_name)

        # compara con la lista y pone 1 si esta
        test = np.isin(data, matching_list).astype(int)
        # suma acumulada a traves de cada experimento
        test = np.cumsum(test, axis=1)
        # promedio de todos los experimentos
        test = np.average(test, axis=0)

        ax.plot(range(self.num_episodes), test, label=var_name)
        return fig, ax

    def get_pie_plot(self, var_name, mapping_dict):
        """
        Cuenta los elementos de una lista y los agrupa segun mapping_dict. Si alguna falta
        le asigna la etiqueta other.
        :param var_name:
        :param mapping_dict:
        :return:
        """

        fig, ax = plt.subplots()
        data = self.get_var_from_data(var_name)
        data = data.astype(int).flatten().tolist()
        data_copy = data[:]

        labels = mapping_dict.keys()

        count = []
        for l in labels:
            c = 0
            for d in range(len(data)):
                if data[d] in mapping_dict[l]:
                    c += 1
                    data_copy.remove(data[d])
            count.append(c)

        labels.append('other')
        count.append(len(data_copy))

        ax.pie(count, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        return fig, ax

    def save_data(self, name):
        numpy_data = np.array(self.data)
        np.save(name, numpy_data)
        return self

    @staticmethod
    def load_data(name, var_name_list=None):
        """
        Carga un archivo de datos y crea un objeto LinesPlotter
        :param name: el nombre del archivo que contiene los datos guardado con save_data()
        :param var_name_list: el nombre de los datos guardados. Si no esta les asigna un entero.
        :return:
        """
        data = np.load(name)
        num_experiments = data.shape[0]
        num_episodes = data.shape[1]

        if var_name_list is None:
            var_name_list = [str(i) for i in range(data.shape[2])]
        elif len(var_name_list) != data.shape[2]:
            raise Exception('Invalid var_name_list. Must have len'+str(data.shape[2]))

        plotter = LinesPlotter(var_name_list, num_experiments, num_episodes)
        plotter.data = data
        return plotter
