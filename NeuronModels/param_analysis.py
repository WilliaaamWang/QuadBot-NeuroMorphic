import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from mqif_neuron import MQIFNeuron as NEURON, single_simulation, detect_burst
# from synaptic_neuron import SynapticNeuron as NEURON, simulate_neuron
# from half_centre import simulate_halfcentre

def calculate_interburst_frequency(interburst_interval):
    if len(interburst_interval) == 0:
        return np.nan  # Use NaN when there's no burst
    else:
        # Example: frequency = 1/mean interval
        return 1.0 / np.mean(interburst_interval)
    
def calculate_intraburst_frequency(intraburst_interval):
    if len(intraburst_interval) == 0:
        return np.nan  # Use NaN when there's no burst
    else:
        return 1.0 / np.mean(intraburst_interval)
    
def calculate_spikes_per_burst(bursts):
    if len(bursts) <= 1:
        return np.nan
    else:
        return len(bursts[1])

def two_parameter_sweep(param1, p1values, param2, p2values, I_ext_array, dt=1e-4, runtime=10, plotter=False):
    """
    Configure neurons with different parameters and run them with the same I_ext_array.
    param: string of the parameter to sweep
    pvalues: array of parameter values
    I_ext_array: array of I_ext values
    dt: time step
    runtime: simulation time
    plotter: boolean to plot the results
    """

    interburst_filename = f"interburst_{p1values[0]:.2f}_{p1values[-1]:.2f}_{p2values[0]:.2f}_{p2values[-1]:.2f}.pkl"
    intraburst_filename = f"intraburst_{p1values[0]:.2f}_{p1values[-1]:.2f}_{p2values[0]:.2f}_{p2values[-1]:.2f}.pkl"
    spikes_per_burst_filename = f"spikes_per_burst_{p1values[0]:.2f}_{p1values[-1]:.2f}_{p2values[0]:.2f}_{p2values[-1]:.2f}.pkl"

    current_path = os.path.dirname(os.path.abspath(__file__))
    interburst_freq_file = os.path.join(current_path, "sweep", interburst_filename)
    intraburst_freq_file = os.path.join(current_path, "sweep", intraburst_filename)
    spikes_per_burst_file = os.path.join(current_path, "sweep", spikes_per_burst_filename)
    
    
    p1grid, p2grid = np.meshgrid(p1values, p2values, indexing='ij')

    if os.path.exists(interburst_freq_file):
        with open(interburst_freq_file, 'rb') as f:
            interburst_freq_grid = pickle.load(f)
        with open(intraburst_freq_file, 'rb') as f:
            intraburst_freq_grid = pickle.load(f)
        with open(spikes_per_burst_file, 'rb') as f:
            spikes_per_burst_grid = pickle.load(f)
    else:
        # p1sparse, p2sparse = np.meshgrid(p1values, p2values, sparse=True)
        
        interburst_freq_grid = np.zeros_like(p1grid, dtype=float)
        intraburst_freq_grid = np.zeros_like(p1grid, dtype=float)
        spikes_per_burst_grid = np.zeros_like(p1grid, dtype=float)

        for p1 in range(len(p1values)):
            for p2 in range(len(p2values)):
                # neuron = MQIFNeuron()
                # neuron = SynapticNeuron()
                neuron = NEURON()
                # Reconfigure the neuron with the new parameters
                if hasattr(neuron, param1) & hasattr(neuron, param2):
                    setattr(neuron, param1, p1values[p1])
                    setattr(neuron, param2, p2values[p2])
                else:
                    raise AttributeError("Neuron does not have the parameter")
                # Run the simulation
                t_array, V_array, Vs_array, Vus_array, spike_times = single_simulation(
                neuron, I_ext_array, dt=dt, runtime=runtime, plotter=plotter
            )
                # Detect the burst
                bursts, interburst_intervals, intraburst_intervals = detect_burst(spike_times)
                # print((p1, p2), np.mean(interburst_intervals))
                interburst_freq_grid[p1, p2] = calculate_interburst_frequency(interburst_intervals)
                intraburst_freq_grid[p1, p2] = calculate_intraburst_frequency(intraburst_intervals)
                spikes_per_burst_grid[p1, p2] = calculate_spikes_per_burst(bursts)

        with open(interburst_freq_file, 'wb') as f:
            pickle.dump(interburst_freq_grid, f)
        with open(intraburst_freq_file, 'wb') as f:
            pickle.dump(intraburst_freq_grid, f)
        with open(spikes_per_burst_file, 'wb') as f:
            pickle.dump(spikes_per_burst_grid, f)

    plt.figure(figsize=(10, 6))
    interburst_plot = plt.contourf(p1grid, p2grid, interburst_freq_grid, cmap='viridis')
    plt.colorbar(interburst_plot, label='Interburst frequency (Hz)')
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title(f"Interburst frequency for {param1} and {param2}")
    figname = os.path.join(current_path, f"interburst_{p1values[0]:.2f}_{p1values[-1]:.2f}_{p2values[0]:.2f}_{p2values[-1]:.2f}.png")
    plt.savefig(figname)
    plt.show()

    plt.figure(figsize=(10, 6))
    intraburst_plot = plt.contourf(p1grid, p2grid, intraburst_freq_grid, cmap='viridis')
    plt.colorbar(intraburst_plot, label='Intraburst frequency (Hz)')
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title(f"Intraburst frequency for {param1} and {param2}")
    figname = os.path.join(current_path, f"intraburst_{p1values[0]:.2f}_{p1values[-1]:.2f}_{p2values[0]:.2f}_{p2values[-1]:.2f}.png")
    plt.savefig(figname)
    plt.show()

    plt.figure(figsize=(10, 6))
    spikes_per_burst_plot = plt.contourf(p1grid, p2grid, spikes_per_burst_grid, cmap='viridis')
    plt.colorbar(spikes_per_burst_plot, label='Spikes per burst')
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title(f"Spikes per burst for {param1} and {param2}")
    figname = os.path.join(current_path, f"spikes_per_burst_{p1values[0]:.2f}_{p1values[-1]:.2f}_{p2values[0]:.2f}_{p2values[-1]:.2f}.png")
    plt.savefig(figname)
    plt.show()


"""
Parameters: Vs0, Vus0
Features: 
    interburst frequency, 
    intraburst frequency
    burst duration
    spike count
Goal: find combinations of Vs0 and Vus0 that:

"""



def main():
    dt = 1e-4
    runtime = 5
    time_array = np.arange(0, runtime, dt)
    I_ext_array = np.zeros(int(runtime/dt))
    I_ext_array[int(1/dt):] = 5

    # params = {"Vs0": [-55, -52, -50], "Vus0": [-60, -56, -52, -48]}

    param1 = "Vs0"
    param2 = "Vus0"
    # p1values = [-55, -52, -50]
    # p2values = [-60, -56, -52, -48]
    # p1values = np.linspace(-60, -48, 10)
    # p2values = np.linspace(-70, -40, 20)
    p1values = np.linspace(-55, -48, 20)
    p2values = np.linspace(-60, -46, 20)

    two_parameter_sweep(param1, p1values, param2, p2values, I_ext_array, dt, runtime, plotter=False)

if __name__ == "__main__":
    main()