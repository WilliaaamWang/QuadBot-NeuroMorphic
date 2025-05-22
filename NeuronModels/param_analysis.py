# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# import pickle
# from mqif_neuron import MQIFNeuron as NEURON, single_simulation, detect_burst
# from synaptic_neuron import SynapticNeuron as NEURON, simulate_neuron
# from half_centre import simulate_halfcentre

# def calculate_interburst_frequency(interburst_interval):
#     if len(interburst_interval) == 0:
#         return np.nan  # Use NaN when there's no burst
#     else:
#         # Example: frequency = 1/mean interval
#         return 1.0 / np.mean(interburst_interval)
    
# def calculate_intraburst_frequency(intraburst_interval):
#     if len(intraburst_interval) == 0:
#         return np.nan  # Use NaN when there's no burst
#     else:
#         return 1.0 / np.mean(intraburst_interval)
    
# def calculate_spikes_per_burst(bursts):
#     if len(bursts) <= 1:
#         return np.nan
#     else:
#         return len(bursts[1])

# def two_parameter_sweep(param1, p1values, param2, p2values, I_ext_array, dt=1e-4, runtime=10, plotter=False):
#     """
#     Configure neurons with different parameters and run them with the same I_ext_array.
#     param: string of the parameter to sweep
#     pvalues: array of parameter values
#     I_ext_array: array of I_ext values
#     dt: time step
#     runtime: simulation time
#     plotter: boolean to plot the results
#     """

#     interburst_filename = f"interburst_{p1values[0]:.2f}_{p1values[-1]:.2f}_{p2values[0]:.2f}_{p2values[-1]:.2f}.pkl"
#     intraburst_filename = f"intraburst_{p1values[0]:.2f}_{p1values[-1]:.2f}_{p2values[0]:.2f}_{p2values[-1]:.2f}.pkl"
#     spikes_per_burst_filename = f"spikes_per_burst_{p1values[0]:.2f}_{p1values[-1]:.2f}_{p2values[0]:.2f}_{p2values[-1]:.2f}.pkl"

#     current_path = os.path.dirname(os.path.abspath(__file__))
#     interburst_freq_file = os.path.join(current_path, "sweep", interburst_filename)
#     intraburst_freq_file = os.path.join(current_path, "sweep", intraburst_filename)
#     spikes_per_burst_file = os.path.join(current_path, "sweep", spikes_per_burst_filename)
    
    
#     p1grid, p2grid = np.meshgrid(p1values, p2values, indexing='ij')

#     if os.path.exists(interburst_freq_file):
#         with open(interburst_freq_file, 'rb') as f:
#             interburst_freq_grid = pickle.load(f)
#         with open(intraburst_freq_file, 'rb') as f:
#             intraburst_freq_grid = pickle.load(f)
#         with open(spikes_per_burst_file, 'rb') as f:
#             spikes_per_burst_grid = pickle.load(f)
#     else:
#         # p1sparse, p2sparse = np.meshgrid(p1values, p2values, sparse=True)
        
#         interburst_freq_grid = np.zeros_like(p1grid, dtype=float)
#         intraburst_freq_grid = np.zeros_like(p1grid, dtype=float)
#         spikes_per_burst_grid = np.zeros_like(p1grid, dtype=float)

#         for p1 in range(len(p1values)):
#             for p2 in range(len(p2values)):
#                 # neuron = MQIFNeuron()
#                 # neuron = SynapticNeuron()
#                 neuron = NEURON()
#                 # Reconfigure the neuron with the new parameters
#                 if hasattr(neuron, param1) & hasattr(neuron, param2):
#                     setattr(neuron, param1, p1values[p1])
#                     setattr(neuron, param2, p2values[p2])
#                 else:
#                     raise AttributeError("Neuron does not have the parameter")
#                 # Run the simulation
#                 t_array, V_array, Vs_array, Vus_array, spike_times = single_simulation(
#                 neuron, I_ext_array, dt=dt, runtime=runtime, plotter=plotter
#             )
#                 # Detect the burst
#                 bursts, interburst_intervals, intraburst_intervals = detect_burst(spike_times)
#                 # print((p1, p2), np.mean(interburst_intervals))
#                 interburst_freq_grid[p1, p2] = calculate_interburst_frequency(interburst_intervals)
#                 intraburst_freq_grid[p1, p2] = calculate_intraburst_frequency(intraburst_intervals)
#                 spikes_per_burst_grid[p1, p2] = calculate_spikes_per_burst(bursts)

#         with open(interburst_freq_file, 'wb') as f:
#             pickle.dump(interburst_freq_grid, f)
#         with open(intraburst_freq_file, 'wb') as f:
#             pickle.dump(intraburst_freq_grid, f)
#         with open(spikes_per_burst_file, 'wb') as f:
#             pickle.dump(spikes_per_burst_grid, f)

#     plt.figure(figsize=(10, 6))
#     interburst_plot = plt.contourf(p1grid, p2grid, interburst_freq_grid, cmap='viridis')
#     plt.colorbar(interburst_plot, label='Interburst frequency (Hz)')
#     plt.xlabel(param1)
#     plt.ylabel(param2)
#     plt.title(f"Interburst frequency for {param1} and {param2}")
#     figname = os.path.join(current_path, f"interburst_{p1values[0]:.2f}_{p1values[-1]:.2f}_{p2values[0]:.2f}_{p2values[-1]:.2f}.png")
#     plt.savefig(figname)
#     plt.show()

#     plt.figure(figsize=(10, 6))
#     intraburst_plot = plt.contourf(p1grid, p2grid, intraburst_freq_grid, cmap='viridis')
#     plt.colorbar(intraburst_plot, label='Intraburst frequency (Hz)')
#     plt.xlabel(param1)
#     plt.ylabel(param2)
#     plt.title(f"Intraburst frequency for {param1} and {param2}")
#     figname = os.path.join(current_path, f"intraburst_{p1values[0]:.2f}_{p1values[-1]:.2f}_{p2values[0]:.2f}_{p2values[-1]:.2f}.png")
#     plt.savefig(figname)
#     plt.show()

#     plt.figure(figsize=(10, 6))
#     spikes_per_burst_plot = plt.contourf(p1grid, p2grid, spikes_per_burst_grid, cmap='viridis')
#     plt.colorbar(spikes_per_burst_plot, label='Spikes per burst')
#     plt.xlabel(param1)
#     plt.ylabel(param2)
#     plt.title(f"Spikes per burst for {param1} and {param2}")
#     figname = os.path.join(current_path, f"spikes_per_burst_{p1values[0]:.2f}_{p1values[-1]:.2f}_{p2values[0]:.2f}_{p2values[-1]:.2f}.png")
#     plt.savefig(figname)
#     plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pathlib
import pickle
from tqdm import tqdm
# from synapticNeuronClass import SynapticNeuron
from half_centre import SynapticNeuron, simulate_halfcentre
from utils import split_into_bursts, burst_metrics

def run_once(dt, runtime, I_ext_arr, **neuron_kwargs):
    # Use SynapticNeuron instead of MQIFNeuron.
    # Ensure that 'excitatory_Vin' and 'inhibitory_Vin' are provided in neuron_kwargs.
    neuron = SynapticNeuron(**neuron_kwargs)
    t_array, V_trace, _, _, spike_times = simulate_halfcentre(neuron, I_ext_arr, dt, runtime, plotter=False)
    bursts, _ = split_into_bursts(spike_times)
    bm = burst_metrics(bursts)

    # Aggregate metrics: average inter-burst and intra-burst frequencies,
    # and average spikes per burst.
    interburst_hz = np.mean(bm["inter_burst_freq"]) if bm["inter_burst_freq"].size > 0 else np.nan
    if bm["intra_burst_freq_lists"]:
        try:
            intra_vals = np.concatenate(bm["intra_burst_freq_lists"])
            intraburst_hz = np.mean(intra_vals) if intra_vals.size > 0 else np.nan
        except ValueError:
            intraburst_hz = np.nan
    else:
        intraburst_hz = np.nan
    spikes_per_burst = np.mean(bm["n_spikes_per_burst"]) if bm["n_spikes_per_burst"].size > 0 else np.nan

    mode = "bursting" if len(bursts) > 1 else "nonburst"
    return dict(mode=mode,
                interburst_hz=interburst_hz,
                intraburst_hz=intraburst_hz,
                spikes_per_burst=spikes_per_burst)

# ------------------------------------------------------------------ #
def sweep_2d(param1, v1, param2, v2, *,
             dt=1e-4, runtime=5.0, I_ext_amp=5.0,
             cache=True, cachepath=True):
    """
    Sweep two neuron parameters over given ranges.
    param1/param2: str          (attribute names)
    v1/v2: 1-D arrays           (values to sweep)
    """
    cache_path = pathlib.Path(os.path.join(os.path.dirname(__file__), "cache"))
    cache_path.mkdir(exist_ok=True)
    tag = f'{param1}_{v1[0]}_{v1[-1]}_{param2}_{v2[0]}_{v2[-1]}_{len(v1)}x{len(v2)}'
    pkl_file = cache_path / f'{tag}.pkl'
    if cache and pkl_file.exists():
        return pd.read_pickle(pkl_file)

    # Create external input: a step input starting at 1 second.
    I_ext_arr = np.zeros(int(runtime/dt))
    I_ext_arr[int(1.0/dt):] = I_ext_amp

    rows = []
    for a in tqdm(v1, desc=param1):
        for b in v2:
            result = run_once(dt, runtime, I_ext_arr, **{param1: a, param2: b})
            rows.append({param1: a, param2: b, **result})

    df = pd.DataFrame(rows)
    if cache:
        df.to_pickle(pkl_file)
    return df

# ------------------------------------------------------------------ #
def plot_heat(df, x, y, z, title='', mask_nonburst=True):
    sub = df.copy()
    if mask_nonburst:
        sub = sub[sub['mode'] == 'bursting']
    xv = np.sort(sub[x].unique())
    yv = np.sort(sub[y].unique())
    Z = np.full((len(yv), len(xv)), np.nan)
    for _, row in sub.iterrows():
        ix = np.where(xv == row[x])[0][0]
        iy = np.where(yv == row[y])[0][0]
        Z[iy, ix] = row[z]
    X, Y = np.meshgrid(xv, yv)
    plt.figure(figsize=(8, 5))
    cs = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(cs, label=z)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title or f'{z} vs {x}/{y}')
    plt.show()

# ------------------------------------------------------------------ #
if __name__ == '__main__':
    # Example sweep using Vs0 and Vus0 parameters.
    # Also ensure that excitatory_Vin and inhibitory_Vin are provided.
    # Here, dummy inputs are provided as zeros matching a single value.
    dummy_excitatory = np.array([0])
    dummy_inhibitory = np.array([0])
    p1 = np.linspace(-55, -48, 20)
    p2 = np.linspace(-60, -46, 20)
    df = sweep_2d('Vs0', p1, 'Vus0', p2, dt=1e-4, runtime=5, I_ext_amp=5,
                #   excitatory_Vin=dummy_excitatory, 
                #   inhibitory_Vin=dummy_inhibitory
    )

    # Save results for later use.
    df.to_csv(os.path.join(os.path.dirname(__file__), 'Vs0_Vus0.csv'), index=False)

    # Quick visualizations.
    plot_heat(df, 'Vs0', 'Vus0', 'interburst_hz', title='Inter-burst Frequency (Hz)')
    plot_heat(df, 'Vs0', 'Vus0', 'intraburst_hz', title='Intra-burst Frequency (Hz)')
    plot_heat(df, 'Vs0', 'Vus0', 'spikes_per_burst', title='Spikes per Burst')
