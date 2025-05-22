import numpy as np
import matplotlib.pyplot as plt
import os
# import copy

from .synaptic_neuron import SynapticNeuron  # make sure synaptic_neuron.py is available in the same directory

# def sigmoid(x: np.array) -> np.array:
#     result = 1 / (1 + np.exp(-x))
#     return result

# def forward_euler(y, dt, dydt):
#     y_ret = y + dydt*dt
#     return y_ret

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
def simulate_halfcentre(neuronA: SynapticNeuron, neuronB: SynapticNeuron, I_ext_array_A: list, I_ext_array_B: list, excit_ext_A, inhib_ext_A, excit_ext_B, inhib_ext_B, dt = 1e-4, runtime = 5.0, plotter=False, same_start=True):

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
        fig.suptitle(f"Half Centre. Veth A = {neuronA.Ve_threshold}, Vith A = {neuronA.Vi_threshold}, Veth B = {neuronB.Ve_threshold}, Vith B = {neuronB.Vi_threshold}", fontsize=16)
        plt.subplots_adjust(top=0.92)
        plt.savefig(os.path.join(os.path.dirname(__file__),"Halfcentre_Plots", f"synapse_{runtime}_{neuronA.Ve_threshold}_{neuronA.Vi_threshold}_{neuronB.Ve_threshold}_{neuronB.Vi_threshold}.png"))
        # plt.show()

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

    return neuronA, neuronB


    # neuronA = SynapticNeuron(excitatory_Vin=VA_ext[0], inhibitory_Vin=np.array([VA_ext[1], VofB]))
    # neuronB = SynapticNeuron(excitatory_Vin=VB_ext[0], inhibitory_Vin=np.array([VB_ext[1], VofA]))

# def simulate_neurons(current_ext, dt, time):
#     neuronA = SynapticNeuron(Ve_threshold=-20, Vi_threshold=-20, excitatory_Vin=excit_ext_A, inhibitory_Vin=inhib_ext_A)
#     neuronB = SynapticNeuron(Ve_threshold=-20, Vi_threshold=-20, excitatory_Vin=excit_ext_B, inhibitory_Vin=inhib_ext_B)

def main():
    dt = 5e-5
    # dt = 1e-4
    runtime = 10.0

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

    initial_excit = -52
    initial_inhib = -52

    neuronA = SynapticNeuron(excitatory_Vin=initial_excit, inhibitory_Vin=initial_inhib, Ve_threshold=-50, Vi_threshold=-20,)
    neuronB = SynapticNeuron(excitatory_Vin=initial_excit, inhibitory_Vin=initial_inhib, Ve_threshold=-50, Vi_threshold=-30,)


    simulate_halfcentre(neuronA, neuronB, current_ext, current_ext, excit_ext_A, excit_ext_B, inhib_ext_A, inhib_ext_B, dt, runtime, plotter, same_start)

    # print(np.sum(neuronA.I_inhibitory_values))

    # Save neurons' Vvalues to csv
    csv_dir = os.path.join(os.path.dirname(__file__), "Halfcentre_Plots")
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    neuronA_csv = os.path.join(csv_dir, f"neuronA_{runtime}s_{neuronA.Ve_threshold}_{neuronA.Vi_threshold}_I={amplitude}.csv")
    neuronB_csv = os.path.join(csv_dir, f"neuronB_{runtime}s_{neuronB.Ve_threshold}_{neuronB.Vi_threshold}_I={amplitude}.csv")
    np.savetxt(neuronA_csv, neuronA.Vvalues, delimiter=",")
    np.savetxt(neuronB_csv, neuronB.Vvalues, delimiter=",")


if __name__ == "__main__":
    main()
    