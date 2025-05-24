import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from utils import extract_features, detect_spikes
# import copy

if __package__ is None or __package__ == "":
    import os, sys
    sys.path.append(os.path.dirname(__file__))
    from synaptic_neuron import SynapticNeuron
else:
    from .synaptic_neuron import SynapticNeuron


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
        fig, axs = plt.subplots(5,1, figsize=(12, 8))

        axs[0].plot(t_array, I_ext_array_A, color="tab:blue", label="Applied I_ext")
        axs[0].plot(t_array, I_ext_array_B, color="tab:orange")
        axs[0].set_title("Applied I_ext")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("I_ext (mA/nF)")
        axs[0].legend()

        axs[1].plot(t_array, neuronA.Vvalues, color="tab:green")
        axs[1].set_title("Neuron A Membrane Potential (V)")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Voltage (mV)")

        axs[3].plot(t_array, neuronB.Vvalues, color="tab:red")
        axs[3].set_title("Neuron B Membrane Potential (V)")
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("Voltage (mV)")

        axs[2].plot(t_array, neuronA.Sivalues, color="tab:blue", label="Neuron A Si")
        axs[2].set_title("Neuron A Si")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Synaptic Weight")

        axs[4].plot(t_array, neuronB.Sivalues, color="tab:orange", label="Neuron B Si")
        axs[4].set_title("Neuron B Si")
        axs[4].set_xlabel("Time (s)")
        axs[4].set_ylabel("Synaptic Weight")

        plt.tight_layout()
        fig.suptitle(f"Half Centre. Veth A = {neuronA.Ve_threshold}, Vith A = {neuronA.Vi_threshold}, Veth B = {neuronB.Ve_threshold}, Vith B = {neuronB.Vi_threshold}", fontsize=16)
        plt.subplots_adjust(top=0.92)
        plt.savefig(os.path.join(os.path.dirname(__file__),"Halfcentre_Plots", f"halfcentre_{runtime}_{neuronA.Ve_threshold}_{neuronA.Vi_threshold}_{neuronB.Ve_threshold}_{neuronB.Vi_threshold}_dt={dt}.png"))
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


def save_hc_data(neuronA, neuronB, dt, filename):
    """
    Save neuron data to a CSV file.
    """
    data = {
        "Time": np.arange(len(neuronA.Vvalues)) * dt,
        "Neuron A V": neuronA.Vvalues,
        "Neuron B V": neuronB.Vvalues,
        # "Neuron A Se": neuronA.Se,
        # "Neuron B Se": neuronB.Se,
        # "Neuron A Si": neuronA.Si,
        # "Neuron B Si": neuronB.Si
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Analyse simulated half-centre features from saved csv data
def analyse_hc_features(dt, filename):
    """
    Analyse features of the half-centre model from a CSV file.
    """
    data = pd.read_csv(filename)
    time_axis = data["Time"].values
    neuronA_V = data["Neuron A V"].values
    neuronB_V = data["Neuron B V"].values

    # Extract all features from the neurons
    features_A = extract_features(neuronA_V, dt)
    features_B = extract_features(neuronB_V, dt)

    # Save features to CSV
    features_A_df = pd.DataFrame(features_A)
    features_B_df = pd.DataFrame(features_B)
    features_A_df.to_csv(filename.replace(".csv", "_A_features.csv"), index=False)
    features_B_df.to_csv(filename.replace(".csv", "_B_features.csv"), index=False)

    # Print features
    peaks_A = features_A["peaks"]
    peaks_B = features_B["peaks"]
    print(f"Neuron A Regime: {features_A['regime']}")
    print(f"Neuron B Regime: {features_B['regime']}")
    print(f"Neuron A Spike count: {features_A['spike_count']}")
    print(f"Neuron B Spike count: {features_B['spike_count']}")
    print(f"Neuron A Mean ISI: {features_A['mean_isi']:.4f} s")
    print(f"Neuron B Mean ISI: {features_B['mean_isi']:.4f} s")
    print(f"Neuron A CV ISI: {features_A['cv_isi']:.4f}")
    print(f"Neuron B CV ISI: {features_B['cv_isi']:.4f}")
    print(f"Neuron A Number of bursts: {features_A['n_bursts']}")
    print(f"Neuron B Number of bursts: {features_B['n_bursts']}")
    # print(f"Neuron A: {features_A['burst_freq']:.4f} Hz")
    # print(f"Neuron B: {features_B['burst_freq']:.4f} Hz")
    print(f"Neuron A Mean spikes per burst: {features_A['mean_spikes_per_burst']:.4f}")
    print(f"Neuron B Mean spikes per burst: {features_B['mean_spikes_per_burst']:.4f}")
    # print(f"Neuron A: {features_A['mean_burst_duration']:.4f} s")
    # print(f"Neuron B: {features_B['mean_burst_duration']:.4f} s")
    print(f"Neuron A Duty cycle: {features_A['duty_cycle']:.4f}")
    print(f"Neuron B Duty cycle: {features_B['duty_cycle']:.4f}")
    print(f"Neuron A Interburst frequency: {features_A['interburst_freq']:.4f}")
    print(f"Neuron B Interburst frequency: {features_B['interburst_freq']:.4f}")
    print(f"Neuron A Intraburst frequency: {features_A['intraburst_freq']}")
    print(f"Neuron B Intraburst frequency: {features_B['intraburst_freq']}")


    # Detect spikes
    # peaks_A, t_spike_A = detect_spikes(neuronA_V, fs=1/dt, threshold=-10.0, refractory_ms=2.0, prominence=5.0)
    # peaks_B, t_spike_B = detect_spikes(neuronB_V, fs=1/dt, threshold=-10.0, refractory_ms=2.0, prominence=5.0)

    # # Split into bursts
    # isi_gap_A = choose_isi_gap(t_spike_A)
    # isi_gap_B = choose_isi_gap(t_spike_B)
    
    # bursts_A, isi_gap_A = split_into_bursts(t_spike_A, isi_gap=isi_gap_A)
    # bursts_B, isi_gap_B = split_into_bursts(t_spike_B, isi_gap=isi_gap_B)

    # # Calculate burst metrics
    # burst_metrics_data_A = burst_metrics(bursts_A)
    # burst_metrics_data_B = burst_metrics(bursts_B)

    # print(f"ISI gap A: {isi_gap_A:.4f} s")
    # print(f"ISI gap B: {isi_gap_B:.4f} s")
    
    # print(f"Spikes per burst A: {burst_metrics_data_A['n_spikes_per_burst']}")
    # print(f"Spikes per burst B: {burst_metrics_data_B['n_spikes_per_burst']}")
    
    # print(f"Inter-burst frequency A: {burst_metrics_data_A['inter_burst_freq']}")
    # print(f"Inter-burst frequency B: {burst_metrics_data_B['inter_burst_freq']}")
    
    # print(f"Intra-burst frequencies A: {burst_metrics_data_A['intra_burst_freq_lists']}")
    # print(f"Intra-burst frequencies B: {burst_metrics_data_B['intra_burst_freq_lists']}")

def demo(mode="both"):
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

    csv_dir = os.path.join(os.path.dirname(__file__), "Halfcentre_Plots")

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    filename = os.path.join(csv_dir, f"halfcentre_{runtime}_{neuronA.Ve_threshold}_{neuronA.Vi_threshold}_{neuronB.Ve_threshold}_{neuronB.Vi_threshold}_I={amplitude}_dt={dt}.csv")

    if mode == "both":
        # Simulate the half-centre synapse model
        neuronA, neuronB = simulate_halfcentre(neuronA, neuronB, current_ext, current_ext, excit_ext_A, excit_ext_B, inhib_ext_A, inhib_ext_B, dt, runtime, plotter, same_start)    
        
        # Save neuron data to CSV
        save_hc_data(neuronA, neuronB, dt, filename)
        # Analyse features from saved data
        analyse_hc_features(dt, filename)

    elif mode == "sim":
        # Simulate the half-centre synapse model
        neuronA, neuronB = simulate_halfcentre(neuronA, neuronB, current_ext, current_ext, excit_ext_A, excit_ext_B, inhib_ext_A, inhib_ext_B, dt, runtime, plotter, same_start)

        # Save neuron data to CSV
        save_hc_data(neuronA, neuronB, dt, filename)

    elif mode == "analyze":
        # Analyse features from saved data
        csvname = "halfcentre_10.0_-50_-20_-50_-30_I=5_dt=0.0001"
        filename = os.path.join(csv_dir, csvname + ".csv")
        analyse_hc_features(dt, filename)
    

if __name__ == "__main__":
    mode = "both"
    # mode = "sim"
    # mode = "analyze"
    demo()
    