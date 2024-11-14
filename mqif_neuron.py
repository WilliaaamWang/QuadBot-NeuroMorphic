import numpy as np

class MQIFNeuron:
    def __init__(self, neuron_id, v_rest=-65.0, v_reset=-68.0, v_threshold=-50.0, i_ext=0.0, g_syn=0.0, tau_syn=10.0):
        self.id = neuron_id #unique identifier for each neuron in the network, A, B, C, etc.
        self.v = v_rest #v: membrane potential of the neuron, v_rest: initial resting membrane potential of the neuron
        self.v_rest = v_rest 
        self.v_reset = v_reset #v_reset: reset potential of the neuron (back to this value after spikes)
        self.v_threshold = v_threshold #v_threshold: When the membrane potential reaches or exceeds this value, the neuron fires
        self.i_ext = i_ext #i_ext: External current input to the neuron
        self.g_syn = g_syn #g_syn: Maximum conductance of synapses connected to this neuron, how strongly the synapse can impact the neuron
        self.tau_syn = tau_syn #tau_syn: Time constant for synaptic current decay
        self.has_spiked = False
        self.spike_times = []
        
    #Returns the external input current at time t (in ms).
    def get_external_input(self, t):
        return self.i_ext


        """
        Update the membrane potential using the MQIF model.       
        Parameters:
        dt: Time step in ms
        i_syn: Synaptic input current (nA)
        """
    def update(self, dt, i_syn):
        #i_syn: synaptic input current coming from other neurons
        dv = dt * (- (self.v - self.v_rest) + i_syn + self.get_external_input(dt))
        self.v += dv
        
        # Check if neuron spikes
        if self.v >= self.v_threshold:
            self.has_spiked = True
            self.spike_times.append(dt)
            self.v = self.v_reset  # Reset after spike
        else:
            self.has_spiked = False
