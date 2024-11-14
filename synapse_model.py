class Synapse:
    #pre_neuron: sends signal, post_neuron: receives signal, g_syn: synaptic conductance(strength of connection between neurons), tau_syn: time cosntant for the synaptic spike to decay
    def __init__(self, pre_neuron, post_neuron, g_syn, tau_syn):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.g_syn = g_syn
        self.tau_syn = tau_syn

    #Calculate the synaptic current for the post-neuron.
    def get_synaptic_current(self):
        if self.pre_neuron.has_spiked:
            return self.g_syn
        return 0.0
