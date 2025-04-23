import numpy as np
import matplotlib.pyplot as plt
import os
# import copy
# from synaptic_neuron import SynapticNeuron

def sigmoid(x: np.array) -> np.array:
    result = 1 / (1 + np.exp(-x))
    return result

def forward_euler(y, dt, dydt):
    y_ret = y + dydt*dt
    return y_ret

class SynapticNeuron:
    def __init__(
        self,
        V0 = -52, Vs0 = -50, Vus0 = -52,
        tau_s = 4.3, tau_us = 278,
        g_f = 1.0, g_s = 0.5, g_us = 0.015,
        V_threshold = 20, V_peak = 20,
        V_reset = -45, Vs_reset = 7.5,
        delta_Vus = 1.7,
        # Synaptic parameters
        Ve0 = 0, Vi0 = -90,
        Ve_threshold = -20, Vi_threshold = -20,
        # g_syn_e = 200, g_syn_i = 200,
        g_syn_e = 0.5, g_syn_i = 30,
        tau_e = 50, tau_i = 50,
        cap = 0.82, k = 250.0,
        I_ext=0.0, excitatory_Vin=None, inhibitory_Vin=None,):

        self.V0 = V0
        self.Vs0 = Vs0
        self.Vus0 = Vus0
        self.tau_s = tau_s
        self.tau_us = tau_us
        self.g_f = g_f
        self.g_s = g_s
        self.g_us = g_us
        self.V_threshold = V_threshold
        self.V_reset = V_reset
        self.Vs_reset = Vs_reset
        self.V_peak = V_peak
        self.delta_Vus = delta_Vus
        self.Ve0 = Ve0
        self.Vi0 = Vi0
        self.Ve_threshold = Ve_threshold
        self.Vi_threshold = Vi_threshold
        self.g_syn_e = g_syn_e
        self.g_syn_i = g_syn_i
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.cap = cap
        self.k = k

        # self.I_ext = I_ext
        self.excitatory_Vin = np.atleast_1d(excitatory_Vin) if excitatory_Vin is not None else np.array([-54])
        self.inhibitory_Vin = np.atleast_1d(inhibitory_Vin) if inhibitory_Vin is not None else np.array([-54])


        self.V = V0
        self.Vs = Vs0
        self.Vus = Vus0
        self.Se = np.zeros_like(self.excitatory_Vin.shape)
        self.Si = np.zeros_like(self.inhibitory_Vin.shape)

        self.Vvalues = []
        self.Vsvalues = []
        self.Vusvalues = []
        self.Sevalues = []
        self.Sivalues = []
        self.I_excitatory_values = []
        self.I_inhibitory_values = []

        self.I_ext = I_ext
        self.excitatory_Vin = excitatory_Vin if excitatory_Vin is not None else Ve0
        self.inhibitory_Vin = inhibitory_Vin if inhibitory_Vin is not None else Vi0

    def compute_derivatives(self, log_currents=True):
        """
        Se, Si corresponds to the probability of the synapse being open
        g_synapse = g_syn_e * Se OR g_syn_i * Si
        """

        dVs = self.k * (self.V - self.Vs) / self.tau_s
        dVus = self.k * (self.V - self.Vus) / self.tau_us

        sigmoid_Ve = sigmoid((self.excitatory_Vin - self.Ve_threshold)*40)
        sigmoid_Vi = sigmoid((self.inhibitory_Vin - self.Vi_threshold)* 40)
        dSe = self.k * (sigmoid_Ve - self.Se) / self.tau_e
        dSi = self.k * (sigmoid_Vi - self.Si) / self.tau_i

        I_excitatory = self.g_syn_e * self.Se * (self.V - self.Ve0)
        I_inhibitory = self.g_syn_i * self.Si * (self.V - self.Vi0)
        
        dV = self.k * (self.I_ext - np.sum(I_excitatory) - np.sum(I_inhibitory) \
            + self.g_f*((self.V-self.V0)**2) \
            - self.g_s*((self.Vs-self.Vs0)**2) \
            - self.g_us*((self.Vus-self.Vus0)**2) ) / self.cap

        if log_currents:
            self.I_excitatory_values.append(I_excitatory)
            self.I_inhibitory_values.append(I_inhibitory)

        return dV, dVs, dVus, dSe, dSi
    
    def update_inputs(self, I_ext = None, excitatory_Vin = None, inhibitory_Vin = None):
        if I_ext is not None:
            self.I_ext = I_ext
        if excitatory_Vin is not None:
            self.excitatory_Vin = excitatory_Vin
        if inhibitory_Vin is not None:
            self.inhibitory_Vin = inhibitory_Vin

    def update_state(self, dt):
        dV, dVs, dVus, dSe, dSi = self.compute_derivatives()
        
        # First calculate value of V_new 
        V_new = forward_euler(self.V, dt, dV)

        if V_new >= self.V_threshold:
            # Set neuron voltage to fixed V_peak
            self.V = self.V_peak

            # Recompute new derivatives based on V=V_peak
            dV_spike, dVs_spike, dVus_spike, dSe_spike, dSi_spike = self.compute_derivatives(log_currents=False)

            # # Update values of Vs, Vus, Se, Si
            Vs_new = forward_euler(self.Vs, dt, dVs_spike)
            Vus_new = forward_euler(self.Vus, dt, dVus_spike)
            Se_new = forward_euler(self.Se, dt, dSe_spike)
            Si_new = forward_euler(self.Si, dt, dSi_spike)
            
            # Append peak value V_peak to Vvalues
            self.Vvalues.append(self.V)

            # Reset after spiking action
            self.V = self.V_reset
            self.Vs = self.Vs_reset
            self.Vus += self.delta_Vus
            # self.Se = np.zeros_like(self.excitatory_Vin)
            # self.Si = np.zeros_like(self.inhibitory_Vin)
        else:
            self.V = V_new

            Vs_new = forward_euler(self.Vs, dt, dVs)
            Vus_new = forward_euler(self.Vus, dt, dVus)
            Se_new = forward_euler(self.Se, dt, dSe)
            Si_new = forward_euler(self.Si, dt, dSi)

            self.Vs = Vs_new
            self.Vus = Vus_new
            self.Se = Se_new
            self.Si = Si_new

            self.Vvalues.append(self.V)

        # self.Vsvalues.append(self.Vs)
        # self.Vusvalues.append(self.Vus)
        self.Sevalues.append(self.Se)
        self.Sivalues.append(self.Si)


    """
    def Vs_dot(self):
        return self.k * (self.V - self.Vs) / self.tau_s
    
    def Vus_dot(self):
        return self.k * (self.V - self.Vus) / self.tau_us

    #! For half-centre synapse there are 2 inhibitory inputs for each neuron, make it an array of array
    def Se_dot(self, excitatory_Vin) -> np.ndarray:
        # Non-linear function to control synaptic gating
        sigmoid_Ve = self.sigmoid((excitatory_Vin - self.Ve_threshold)*40)
        return self.k * (sigmoid_Ve - self.Se) / self.tau_e
    
    def Si_dot(self, inhibitory_Vin) -> np.ndarray:
        # Non-linear function to control synaptic gating
        sigmoid_Vi = self.sigmoid((inhibitory_Vin - self.Vi_threshold)* 40)
        return self.k * (sigmoid_Vi - self.Si) / self.tau_i
    
    def V_dot(self, I_ext):
        
        Ie = self.g_syn_e * self.Se * (self.Ve0 - self.V)
        Ii = self.g_syn_i * self.Si * (self.Vi0 - self.V)
        
        return self.k * (I_ext + np.sum(Ie) + np.sum(Ii) + self.g_f*((self.V-self.V0)**2) - self.g_s*((self.Vs-self.Vs0)**2) - self.g_us*((self.Vus-self.Vus0)**2)) / self.cap
    
    def update(self, dt, I_ext):
        new_Se = self.Se + dt * self.Se_dot(self.excitatory_Vin)
        new_Si = self.Si + dt * self.Si_dot(self.inhibitory_Vin)
        new_Vs = self.Vs + dt * self.Vs_dot()
        new_Vus = self.Vus + dt * self.Vus_dot()
        new_V = self.V + dt * self.V_dot(I_ext)
        return new_Se, new_Si, new_Vs, new_Vus, new_V
    """

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
def simulate_synapse(neuronA: SynapticNeuron, neuronB: SynapticNeuron, I_ext_array_A: list, I_ext_array_B: list, excit_ext_A, inhib_ext_A, excit_ext_B, inhib_ext_B, dt = 1e-4, runtime = 5.0, plotter=False, same_start=True):

    # excit_A = np.array(excit_ext_A)
    # excit_B = np.array(excit_ext_B)
    # inhib_A = np.append(inhib_ext_A, -52) #! TODO: assumed initial V=-52
    # inhib_B = np.append(inhib_ext_B, -52) #! TODO: assumed initial V=-52

    # if neuronA is None or neuronB is None:
    #     neuronA = SynapticNeuron(excitatory_Vin=excit_A, inhibitory_Vin=inhib_ext_A)
    #     neuronB = SynapticNeuron(excitatory_Vin=excit_B, inhibitory_Vin=inhib_ext_B)
    
    t_array = np.arange(0, runtime, dt)

    spike_times = []
    cnt = 0

    neuronA.V = neuronA.V0
    if same_start:
        neuronB.V = neuronB.V0
    else:
        neuronB.V = neuronB.V0 + 0.1

    prev_VA = neuronA.V
    prev_VB = neuronB.V
    
    for t, I_ext_A, I_ext_B in zip(t_array, I_ext_array_A, I_ext_array_B):

        excit_A = np.array(excit_ext_A)
        excit_B = np.array(excit_ext_B)
        inhib_A = np.append(inhib_ext_A, prev_VB)
        inhib_B = np.append(inhib_ext_B, prev_VA)
        
        # tempA = copy.deepcopy(neuronA)
        # tempB = copy.deepcopy(neuronB)
        # tempA.update_inputs(I_ext=I_ext_A, excitatory_Vin=excit_A, inhibitory_Vin=inhib_A)
        # tempB.update_inputs(I_ext=I_ext_B, excitatory_Vin=excit_B, inhibitory_Vin=inhib_B)

        # tempA.update_state(dt)
        # tempB.update_state(dt)

        # neuronA.V = tempA.V
        # neuronB.V = tempB.V


        neuronA.update_inputs(I_ext=I_ext_A, excitatory_Vin=excit_A, inhibitory_Vin=inhib_A)
        neuronB.update_inputs(I_ext=I_ext_B, excitatory_Vin=excit_B, inhibitory_Vin=inhib_B)
        
        neuronA.update_state(dt)
        neuronB.update_state(dt)

        prev_VA = neuronA.V
        prev_VB = neuronB.V

        # if cnt % 10000 == 0:
        #     print("Time, Excit_A, Excit_B, Inhib_A, Inhib_B, Se_A, Se_B, Si_A, Si_B")
        #     print(f"Time: {t}, {excit_A}, {excit_B}, {inhib_A}, {inhib_B}, {neuronA.Se}, {neuronB.Se}, {neuronA.Si}, {neuronB.Si}")
        # cnt += 1

    if plotter:
        fig, axs = plt.subplots(4,1, figsize=(12, 8))

        # axs[0, 0].plot(t_array, I_ext_array_A, color="tab:blue")
        # axs[0, 0].set_title("Neuron A I_ext Input")
        # axs[0, 0].set_xlabel("Time (s)")
        # axs[0, 0].set_ylabel("I_ext")

        # axs[1, 0].plot(t_array, I_ext_array_B, color="tab:orange")
        # axs[1, 0].set_title("Neuron B I_ext Input")
        # axs[1, 0].set_xlabel("Time (s)")
        # axs[1, 0].set_ylabel("I_ext")

        axs[0].plot(t_array, neuronA.Vvalues, color="tab:green")
        axs[0].set_title("Neuron A Membrane Potential (V)")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Voltage (mV)")

        axs[2].plot(t_array, neuronB.Vvalues, color="tab:red")
        axs[2].set_title("Neuron B Membrane Potential (V)")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Voltage (mV)")

        axs[1].plot(t_array, neuronA.Sivalues, color="tab:blue", label="Neuron A Si")
        axs[1].set_title("Neuron A Si")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Synaptic Weight")

        axs[3].plot(t_array, neuronB.Sivalues, color="tab:orange", label="Neuron B Si")
        axs[3].set_title("Neuron B Si")
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("Synaptic Weight")

        plt.tight_layout()
        fig.suptitle(f"Half Centre. VeA = {neuronA.Ve_threshold}, ViA = {neuronA.Vi_threshold}, VeB = {neuronB.Ve_threshold}, ViB = {neuronB.Vi_threshold}", fontsize=16)
        plt.subplots_adjust(top=0.92)
        # plt.savefig(os.path.join(os.getcwd(),"NeuronModels/Synapse_Plots", f"synapse_{runtime}_{neuronA.Ve_threshold}_{neuronA.Vi_threshold}_{neuronB.Ve_threshold}_{neuronB.Vi_threshold}.png"))
        plt.show()

        # # Sum over minor arrays for inhibitory values before plotting
        # I_inh_A_sums = [np.sum(val) for val in neuronA.I_inhibitory_values]
        # I_inh_B_sums = [np.sum(val) for val in neuronB.I_inhibitory_values]

        # fig2, axs2 = plt.subplots(4, 1, figsize=(12, 8))

        # axs2[0].plot(t_array, neuronA.Vvalues, color="tab:green")
        # axs2[0].set_title("Neuron A Membrane Potential (V)")
        # axs2[0].set_xlabel("Time (s)")
        # axs2[0].set_ylabel("Voltage (mV)")

        # axs2[1].plot(t_array, I_inh_A_sums, color="tab:blue")
        # axs2[1].set_title("Neuron A I_inhibitory")
        # axs2[1].set_xlabel("Time (s)")
        # axs2[1].set_ylabel("I_inhibitory")

        # axs2[2].plot(t_array, neuronB.Vvalues, color="tab:red")
        # axs2[2].set_title("Neuron B Membrane Potential (V)")
        # axs2[2].set_xlabel("Time (s)")
        # axs2[2].set_ylabel("Voltage (mV)")

        # axs2[3].plot(t_array, I_inh_B_sums, color="tab:orange")
        # axs2[3].set_title("Neuron B I_inhibitory")
        # axs2[3].set_xlabel("Time (s)")
        # axs2[3].set_ylabel("I_inhibitory")

        # plt.tight_layout()
        # plt.show()


    # neuronA = SynapticNeuron(excitatory_Vin=VA_ext[0], inhibitory_Vin=np.array([VA_ext[1], VofB]))
    # neuronB = SynapticNeuron(excitatory_Vin=VB_ext[0], inhibitory_Vin=np.array([VB_ext[1], VofA]))

# def simulate_neurons(current_ext, dt, time):
#     neuronA = SynapticNeuron(Ve_threshold=-20, Vi_threshold=-20, excitatory_Vin=excit_ext_A, inhibitory_Vin=inhib_ext_A)
#     neuronB = SynapticNeuron(Ve_threshold=-20, Vi_threshold=-20, excitatory_Vin=excit_ext_B, inhibitory_Vin=inhib_ext_B)

def main():
    dt = 5e-5
    # dt = 1e-4
    runtime = 5.0
    numsteps = int(runtime/dt)
    amplitude = 5
    current_ext = np.zeros(numsteps)
    start_time = int(0.5/dt)
    current_ext[start_time:] = amplitude
    # excit_ext_A = [-54]
    # inhib_ext_A = [-54]
    # excit_ext_B = [-54]
    # inhib_ext_B = [-54]

    excit_ext_A = []
    inhib_ext_A = []
    excit_ext_B = []  
    inhib_ext_B = []

    plotter = True
    same_start = False

    neuronA = SynapticNeuron(Ve_threshold=-20, Vi_threshold=-20,)
    neuronB = SynapticNeuron(Ve_threshold=-20, Vi_threshold=-30,)
    # neuronA = SynapticNeuron(Ve_threshold=-20, Vi_threshold=-20, excitatory_Vin=excit_ext_A, inhibitory_Vin=inhib_ext_A)
    # neuronB = SynapticNeuron(Ve_threshold=-20, Vi_threshold=-20, excitatory_Vin=excit_ext_B, inhibitory_Vin=inhib_ext_B)

    simulate_synapse(neuronA, neuronB, current_ext, current_ext, excit_ext_A, excit_ext_B, inhib_ext_A, inhib_ext_B, dt, runtime, plotter, same_start)

    # print(np.sum(neuronA.I_inhibitory_values))


if __name__ == "__main__":
    main()
    