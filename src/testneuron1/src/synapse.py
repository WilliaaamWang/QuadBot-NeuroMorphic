import numpy as np
import os
import matplotlib.pyplot as plt
from synaptic_neuron import SynapticNeuron

# Exponential Euler Integration FOR GATING VARIABLES
def gating_expeuler(t, z, dt, z_inf, tau_z):
    """
    tau_z * dz/dt = z_inf(V) - z
    Assuming dt sufficiently small s.t. membrane potential
    V is constant over the interval
    => tau_z(V) and z_inf(V) can be treated as const
    => z(t+dt) = z_inf + (z(t) - z_inf) * exp(-dt/tau_z)
    z: any gating variable in conductance-based model
    """
    t_ret = t + dt
    z_ret = z_inf + (z - z_inf) * np.exp(-dt/tau_z)
    return t_ret, z_ret

# Simulate the half-centre synapse model
def simulate_synapse():
    k=1.0
    neuronA = SynapticNeuron(k)
    neuronB = SynapticNeuron(k)
    