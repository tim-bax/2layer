from neuron import JAXTwoCompartmentalLayer, NeuronConfig


class ExtraLayer:
    def __init__(self, key, n_extra: int, n_inputs: int, config: NeuronConfig):
        self.core = JAXTwoCompartmentalLayer(key, n_extra, n_inputs, config)

    @property
    def w_dend(self):
        return self.core.w_dend

    @w_dend.setter
    def w_dend(self, value):
        self.core.w_dend = value

    @property
    def w_soma(self):
        return self.core.w_soma

    @w_soma.setter
    def w_soma(self, value):
        self.core.w_soma = value

    @property
    def T_p(self):
        return self.core.T_p

    @property
    def alpha(self):
        return self.core.alpha

    @property
    def alpha_s(self):
        return self.core.alpha_s
