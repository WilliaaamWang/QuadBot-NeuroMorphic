"""
Generic network simulator for arbitrary connectivity.
"""
import numpy as np
from .synaptic_neuron import SynapticNeuron


def simulate_network(
    neurons, excitatory_inputs, inhibitory_inputs,
    I_ext_list, dt=1e-4, runtime=5.0
):
    """
    Simulate a network of SynapticNeuron instances.
    neurons: list of SynapticNeuron
    excitatory_inputs: dict neuron_idx -> list of presynaptic neuron indices
    inhibitory_inputs: dict neuron_idx -> list of presynaptic neuron indices
    I_ext_list: list of length runtime/dt lists, each containing external I_ext for each neuron
    Returns: voltage_traces: ndarray (ntime, n_neurons)
    """
    n = len(neurons)
    t_steps = int(runtime / dt)
    V_traces = np.zeros((t_steps, n))

    for i, neuron in enumerate(neurons):
        neuron.Vvalues = []

    for step in range(t_steps):
        t = step * dt
        # collect previous voltages
        Vs = [neuron.V for neuron in neurons]
        # update inputs for each neuron
        for i, neuron in enumerate(neurons):
            excit = np.array([Vs[j] for j in excitatory_inputs.get(i, [])])
            inhib = np.array([Vs[j] for j in inhibitory_inputs.get(i, [])])
            Iext = I_ext_list[step][i]
            neuron.update_inputs(Iext, excit, inhib)
        # update state
        for i, neuron in enumerate(neurons):
            neuron.update_state(dt)
            V_traces[step, i] = neuron.V
    return V_traces
