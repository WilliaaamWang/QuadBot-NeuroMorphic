import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

def forward_euler(y, dt, dydt):
    y_ret = y + dydt*dt
    return y_ret

class MQIFNeuron:
    def __init__(
            self,
            # Neuron parameters
            V0=-52, Vs0=-50, Vus0=-52,
            tau_s=4.3, tau_us=278,
            g_f=1.0, g_s=0.5, g_us=0.015,
            V_threshold=20, V_reset=-45, Vs_reset=7.5, delta_Vus=1.7,
            # Synaptic inputs
            Ve0=0, Vi0=-90,
            Ve_threshold=-40, Vi_threshold=-40,
            g_syn_e=0.5, g_syn_i=0.5,
            tau_e=1, tau_i=1,
            cap=0.82, k=250.0,
            I_ext=0.0, Ve=None, Vi=None):
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
        
        # Initialise state varaibles at equilibrium value
        self.V = V0
        self.Vs = Vs0
        self.Vus = Vus0
        self.Vvalues = []
        self.Vsvalues = []
        self.Vusvalues = []

        self.has_spiked = False

        self.I_ext = I_ext
        # Excitatory memb potential effects
        self.Ve = Ve if Ve is not None else Ve0
        self.Vi = Vi if Vi is not None else Vi0

    def compute_derivatives(self):
        """
        Return the derivatives of each state variable.
        """

        dVs = self.k * (self.V - self.Vs) / self.tau_s
        dVus = self.k * (self.V - self.Vus) / self.tau_us
        
        # TODO If you want to incorporate synaptic currents, you could do:
        # syn_current = g_syn_e * (self.Ve - V) + g_syn_i * (self.Vi - V)
        # Then add syn_current in the expression for dV
        dV = (self.k / self.cap) * (
            self.I_ext
            + self.g_f * (self.V - self.V0)**2
            - self.g_s * (self.Vs - self.Vs0)**2
            - self.g_us * (self.Vus - self.Vus0)**2
            # + syn_current  # if needed
        )
        
        return dV, dVs, dVus
    
    def update_inputs(self, I_ext = None, Ve = None, Vi = None):
        if I_ext is not None:
            self.I_ext = I_ext
        if Ve is not None:
            self.Ve = Ve
        if Vi is not None:
            self.Vi = Vi

    def update_state(self, dt):
        """
        Update the state of the neuron.
        """
        dV, dVs, dVus = self.compute_derivatives()
        V_new = forward_euler(self.V, dt, dV)
        Vs_new = forward_euler(self.Vs, dt, dVs)
        Vus_new = forward_euler(self.Vus, dt, dVus)
        # V_new = self.V + dt * dV
        # Vs_new = self.Vs + dt * dVs
        # Vus_new = self.Vus + dt * dVus

        # Update membrane potentials
        if V_new >= self.V_threshold:
            self.V = self.V_reset
            self.Vs = self.Vs_reset
            self.Vus += self.delta_Vus
            self.has_spiked = True
        else:
            self.V = V_new
            self.Vs = Vs_new
            self.Vus = Vus_new
            self.has_spiked = False
        
        self.Vvalues.append(self.V)
        self.Vsvalues.append(self.Vs)
        self.Vusvalues.append(self.Vus)

# Simulation
def single_simulation(neuron, I_ext_array, dt=1e-4, runtime=10, plotter=False):
    if neuron is None:
        neuron = MQIFNeuron(tau_s=4.3)

    t_array = np.arange(0, runtime, dt)
    
    spike_times = []
    for t, I_ext in zip(t_array, I_ext_array):
        neuron.update_inputs(I_ext=I_ext)
        neuron.update_state(dt)
        if neuron.has_spiked:
            spike_times.append(t)

    if plotter:
        plt.plot(t_array, neuron.Vvalues)
        plt.xlabel('Time (s)')
        plt.ylabel('Membrane potential (mV)')
        plt.show()
    
    return t_array, neuron.Vvalues, neuron.Vsvalues, neuron.Vusvalues, spike_times

# TODO
# Identifies PARABOLIC BURSTING modulated by Vus0
# Burst detection algorithm -- find the number of bursts and the interburst interval
def detect_burst(spike_times):
    """
    Detects bursts and measures given voltage trace
    (1) Number of spikes per burst event
    (2) Inter-burst frequency
    """
    bursts = []
    current_burst = []
    for i in range(len(spike_times)):
        if len(current_burst) == 0:
            current_burst.append(spike_times[i])
        else:
            if spike_times[i] - current_burst[-1] < 0.1:
                current_burst.append(spike_times[i])
            else:
                bursts.append(current_burst)
                current_burst = [spike_times[i]]
    
    if len(current_burst) > 0:
        bursts.append(current_burst)

    interburst_intervals = []
    for i in range(1, len(bursts)):
        interburst_intervals.append(bursts[i][0] - bursts[i-1][-1])
    
    return bursts, interburst_intervals


def parameter_sweep(param, param_range, I_ext_array, dt=1e-4, runtime=10, plotter=False):
    """
    Sweep over a range of parameter values.
    Return a dictionary or list of results for each parameter value.
    """
    results = []
    
    for value in param_range:
        # Create the neuron with the specified param type
        if param == "Vs0":
            neuron = MQIFNeuron(
                Vs0=value,
            )
        elif param == "Vus0":
            neuron = MQIFNeuron(
                Vus0=value,
            )
        elif param == "tau_s":
            neuron = MQIFNeuron(
                tau_s=value,
            )
        elif param == "tau_us":
            neuron = MQIFNeuron(
                tau_us=value,
            )
        elif param == "g_f":
            neuron = MQIFNeuron(
                g_f=value,
            )
        elif param == "g_s":
            neuron = MQIFNeuron(
                g_s=value,
            )
        elif param == "g_us":
            neuron = MQIFNeuron(
                g_us=value,
            )
        # elif param == 
        
        # Run the simulation
        t_array, V_array, Vs_array, Vus_array, spike_times = single_simulation(
            neuron, I_ext_array, dt=dt, runtime=runtime, plotter=plotter
        )

        bursts, interburst_intervals = detect_burst(spike_times)
        # print(f"{param}={value:.3f}  SpikeCount={len(spike_times)}  BurstCount={len(bursts)}  InterBurstInterval={interburst_intervals}\n")
        
        # Save data
        results.append({
            param: value,
            't_array': t_array,
            'V_array': V_array,
            'Vs': Vs_array,
            'Vus': Vus_array,
            'spike_times': spike_times,
            'bursts': bursts,
            'interburst_intervals': interburst_intervals
        })
    
    # print(results[0].keys())
    return results
    

def sweep_plot(param, param_range, I_ext_array, dt=1e-4, runtime=10, plotter=False):
    
    sweep_results = parameter_sweep(param, param_range, I_ext_array, dt=dt, runtime=runtime, plotter=plotter)

    # Plot voltage traces for each parameter value
    max_subplots = 7
    voltage_traces_per_figure = max_subplots - 1 # 1 subplot for current vs time
    n_plots = len(sweep_results)
    # print([result['spike_times'] for result in sweep_results])

    for group_start in range(0, n_plots, voltage_traces_per_figure):
        group_end = min(group_start + voltage_traces_per_figure, n_plots)
        subset_results = sweep_results[group_start:group_end]

        # 1 for current and n for voltage traces
        n_subplots_group = len(subset_results) + 1

        figsize = (12, 1.6 * n_subplots_group)
        fig, axs = plt.subplots(n_subplots_group, 1, figsize=figsize, sharex=True)
        if n_subplots_group == 2:
            axs = [axs]
        
        # Plot current vs time
        ax_current = axs[0]
        ax_current.plot(subset_results[0]['t_array'], I_ext_array, color='red', label='I_ext')
        ax_current.set_ylabel("Current (mA)")
        ax_current.legend(loc='upper right')

        # Plot voltage traces
        for i, result in enumerate(subset_results):
            ax = axs[i+1]

            ax.plot(result['t_array'], result['V_array'], label=f"{param}={result[param]:.3f}  SpikeCount={len(result['spike_times'])}")

            ax.set_ylabel("Voltage (mV)")
            ax.legend(loc='upper right')

            bursts = result['bursts']
            for burst in bursts:
                burst_mid = np.mean(burst)
                burst_end = burst[-1]
                num_spikes = len(burst)

                # ax.annotate(f"{num_spikes}", (burst_mid, 0), textcoords="offset points", xytext=(0, 10), ha='center')
                # ax.annotate(
                #     f"{num_spikes} spikes",
                #     xy=(burst_mid, -40),
                #     xytext=(burst_mid, -30),
                #     arrowprops=dict(arrowstyle="->", color="blue"),
                #     fontsize=10, color="blue"
                # )
                ax.annotate(
                    f"{num_spikes} spikes",
                    xy=(burst_end, -40),
                    xytext=(burst_end, -30),
                    # arrowprops=dict(arrowstyle="->", color="blue"),
                    fontsize=10, color="blue"
                )

            interburst_intervals = result['interburst_intervals']
            for idx, interburst_interval in enumerate(interburst_intervals):
                start = bursts[idx][-1]
                end = bursts[idx+1][0]
                mid = (start + end) / 2

                ax.annotate(
                    "",
                    xy=(start, -80),
                    xytext=(end, -80),
                    arrowprops=dict(arrowstyle="<->", color="green"),
                )
                ax.text(
                    mid, 
                    -100, 
                    f"{interburst_interval:.3f}", 
                    ha='center',
                    # fontsize=10,
                    color="green")

            # # Only show x-axis label on the last plot
            # if i == n_subplots_group - 1:
            #     ax.set_xlabel("Time (s)")

        
        plt.suptitle(f"Sweep of {param} - Voltage Traces")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        # plt.show()
        # print(group_start, group_end)
        # print(f"{subset_results[0][param]:.3f}_{subset_results[-1][param]:.3f}")
        fig.savefig(f"C:/Cambridge/FYP/Figures/{param}/{param}_{subset_results[0][param]:.3f}to{subset_results[-1][param]:.3f}.png")

    param_list = [result[param] for result in sweep_results]
    st_list = [len(result['spike_times']) for result in sweep_results]
    interburst_intervals_list = [np.mean(result['interburst_intervals']) for result in sweep_results]

    plt_spike = plt.figure()
    plt.plot(param_list, st_list, marker='o')
    plt.xlabel(param)
    plt.ylabel("Spiking times")
    plt.title(f"Spiking times vs {param}")
    # plt.show()
    plt_spike.savefig(f"C:/Cambridge/FYP/Figures/{param}/{param}_spiking_times_{sweep_results[0][param]:.3f}_{sweep_results[-1][param]:.3f}.png")

    plt_inter_freq = plt.figure()
    plt.plot(param_list, interburst_intervals_list, marker='o')
    plt.xlabel(param)
    plt.ylabel("Interburst intervals")
    plt.title(f"Interburst intervals vs {param}")
    # plt.show()
    plt_inter_freq.savefig(f"C:/Cambridge/FYP/Figures/{param}/{param}_interburst_intervals_{sweep_results[0][param]:.3f}_{sweep_results[-1][param]:.3f}.png")



def main():
    dt = 1e-4
    runtime = 10

    t_array = np.arange(0, runtime, dt)
    I_ext_array = np.zeros_like(t_array)
    duty_cycle = 0.5
    amplitude = 5.0
    num_steps = int(runtime / dt)
    start_idx = num_steps // 10
    I_ext_array[start_idx:] = amplitude

    plotter = False

    # single_simulation(None, I_ext_array, dt, runtime, plotter)
    # sweep_results = parameter_sweep(param, param_range, I_ext_array ,dt, runtime, plotter)

    param = "Vs0"
    # param_range = np.linspace(-60, -40, 11)
    param_range = np.arange(-56.5, -48, 1.5)
    sweep_plot(param, param_range, I_ext_array, dt, runtime, plotter)

    param = "Vus0"
    param_range = np.arange(-60, -40, 4)    
    sweep_plot(param, param_range, I_ext_array, dt, runtime, plotter)

    param = "tau_s"
    # param_range = np.arange(10, 20, 1)
    param_range = [0.1, 0.3, 1, 6]
    sweep_plot(param, param_range, I_ext_array, dt, runtime, plotter)

    param = "tau_us"
    param_range = np.arange(200, 800, 100)
    sweep_plot(param, param_range, I_ext_array, dt, runtime, plotter)

    # param = "g_f"
    # param_range = np.arange(0.9, 2, 0.2)
    # sweep_plot(param, param_range, I_ext_array, dt, runtime, plotter)

    # param = "g_s"
    # param_range = np.arange(0.1, 1.1, 0.1)
    # sweep_plot(param, param_range, I_ext_array, dt, runtime, plotter)

    # param = "g_us"
    # param_range = np.arange(0.02, 0.2, 0.02)
    # sweep_plot(param, param_range, I_ext_array, dt, runtime, plotter)

if __name__ == '__main__':
    main()
