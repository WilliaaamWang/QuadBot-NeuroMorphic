import numpy as np
import os
import matplotlib.pyplot as plt
# from synaptic_neuron import SynapticNeuron

class SynapticNeuron:
    def __init__(
        self,
        # I_ext,
        Ve, Vi, 
        Ve_threshold = -40, Vi_threshold = -40,
        Ve0 = 0, Vi0 = -90,
        g_syn_e = 45, g_syn_i = 45,
        tau_e = 1, tau_i = 1,
        V0 = -52,
        Vs0 = -50,
        Vus0 = -52,
        V_threshold = 20,
        V_reset = -45,
        Vs_reset = 7.5,
        delta_Vus = 1.7,
        g_f = 1.0,
        g_s = 0.5,
        g_us = 0.015,
        tau_s = 4.3,
        tau_us = 278,
        C = 0.82,
        k = 250.0,
    ):
        # self.I_ext = I_ext
        # self.Ve = Ve
        # self.Vi = Vi
        self.Ve_threshold = Ve_threshold
        self.Vi_threshold = Vi_threshold
        self.Ve0 = Ve0
        self.Vi0 = Vi0
        self.g_syn_e = g_syn_e
        self.g_syn_i = g_syn_i
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.V0 = V0
        self.Vs0 = Vs0
        self.Vus0 = Vus0
        self.V_threshold = V_threshold
        self.V_reset = V_reset
        self.Vs_reset = Vs_reset
        self.delta_Vus = delta_Vus
        self.g_f = g_f
        self.g_s = g_s
        self.g_us = g_us
        self.tau_s = tau_s
        self.tau_us = tau_us
        self.C = C
        self.k = k

        self.V = V0
        self.Vs = Vs0
        self.Vus = Vus0
        self.Se = np.zeros(len(Ve))
        self.Si = np.zeros(len(Vi))
    
    def sigmoid(self, x):
        result = 1 / (1 + np.exp(-x))
        return result

    def Vs_dot(self):
        return self.k * (self.V - self.Vs) / self.tau_s
    
    def Vus_dot(self):
        return self.k * (self.V - self.Vus) / self.tau_us

    #! For half-centre synapse there are 2 inhibitory inputs for each neuron, make it an array of array
    def Se_dot(self, Ve) -> np.ndarray:
        # Non-linear function to control synaptic gating
        sigmoid_Ve = self.sigmoid((Ve - self.Ve_threshold)*40)
        return self.k * (sigmoid_Ve - self.Se) / self.tau_e
    
    def Si_dot(self, Vi) -> np.ndarray:
        # Non-linear function to control synaptic gating
        sigmoid_Vi = self.sigmoid((Vi - self.Vi_threshold)* 40)
        return self.k * (sigmoid_Vi - self.Si) / self.tau_i
    
    def V_dot(self, I_ext):
        
        Ie = self.g_syn_e * self.Se * (self.Ve0 - self.V)
        Ii = self.g_syn_i * self.Si * (self.Vi0 - self.V)
        
        return self.k * (I_ext + self.g_f*((self.V-self.V0)**2) - self.g_s*((self.Vs-self.Vs0)**2) - self.g_us*((self.Vus-self.Vus0)**2) + np.sum(Ie) + np.sum(Ii)) / self.C
    
    def update(self, dt, I_ext):
        new_Se = self.Se + dt * self.Se_dot(self.Ve)
        new_Si = self.Si + dt * self.Si_dot(self.Vi)
        new_Vs = self.Vs + dt * self.Vs_dot()
        new_Vus = self.Vus + dt * self.Vus_dot()
        new_V = self.V + dt * self.V_dot(I_ext)
        return new_Se, new_Si, new_Vs, new_Vus, new_V

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
    dt = 1e-4
    runtime = 10.0
    numsteps = int(runtime/dt)
    amplitude = 5
    current_ext = np.zeros(numsteps)
    start_index = numsteps // 10
    current_ext[start_index:] = amplitude

    VA_ext = np.array([-54, -54])
    VB_ext = np.array([-54, -54])
    Ve_threshold = -40
    Vi_threshold = -40
    Ve0 = 0
    Vi0 = -90
    g_syn_e = 0.5
    g_syn_i = 0.5
    tau_e = 1
    tau_i = 1
    V_threshold = 20
    V_reset = -45
    Vs_reset = 7.5
    delta_Vus = 1.7
    g_f = 1.0
    g_s = 0.5
    g_us = 0.015
    tau_s = 4.3
    tau_us = 278
    C = 0.82
    k = 250.0
    # Restoration and Regeneration Balance value
    V0 = -52
    Vs0 = -50
    Vus0 = -52

    VofA = V0
    VofB = V0

    neuronA = SynapticNeuron(Ve=VA_ext[0], Vi=np.array([VA_ext[1], VofB]))
    neuronB = SynapticNeuron(Ve=VB_ext[0], Vi=np.array([VB_ext[1], VofA]))
    
    VA_values = []
    VB_values = []
    Si_A_values = []
    Si_B_values = []

    for step in range(numsteps):
        I_ext = current_ext[step]

        # Update gating variables



if __name__ == "__main__":
    simulate_synapse()
    