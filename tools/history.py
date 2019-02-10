class History:
    """
    Clase para llevar los registros de transiciones de un MDP
    """

    def __init__(self):
        self.history = []

    def insert(self, transition_tuple):
        """
        Inserta una nueva transicion (s,a,r,s')
        :param transition_tuple:
        :return:
        """
        if len(transition_tuple) != 4:
            raise Exception('Invalid transition. Required (s,a,r,s')
        self.history.append(transition_tuple)
        return self

    def get_total_reward(self):
        """
        Recupera el total de recompensas recibidas en el historial.
        :return:
        """
        total = 0
        for h in self.history:
            total += h[2]
        return total

    def get_steps_count(self):
        """
        Regresa el numero de transiciones almacenadas.
        :return:
        """
        return len(self.history)

    def get_state_sequence(self):
        """
        Recupera la secuencia de estados.
        :return:
        """
        if len(self.history) < 1:
            return []
        sequence = [self.history[0][0]]
        for s in self.history:
            sequence.append(s[3])
        return sequence

    def clear(self):
        self.history = []
        return self
