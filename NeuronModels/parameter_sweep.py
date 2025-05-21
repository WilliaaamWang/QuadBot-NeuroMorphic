"""
Batch script to sweep key MQIF parameters and extract features in parallel.
"""
import numpy as np
import pandas as pd
import h5py
import itertools
from multiprocessing import Pool
from synapticNeuronClass import SynapticNeuron
from utils import extract_features
import os

# Define parameter ranges
taus0_range = np.linspace(-55.0, -45.0, 5)  # Vs0 values
vus0_range = np.linspace(-60.0, -48.0, 5)   # Vus0 values
Iext_range = np.linspace(5.0, 15.0, 5)      # external currents
gus_range = np.linspace(0.005, 0.03, 3)     # ultraslow gain

def run_sim(params):
    Vs0, Vus0, Iext, g_us = params
    neuron = SynapticNeuron(
        excitatory_Vin=np.array([]),
        inhibitory_Vin=np.array([]),
        Vs0=Vs0,
        Vus0=Vus0,
        g_us=g_us
    )
    dt = 1e-4
    runtime = 10.0
    steps = int(runtime / dt)
    # constant external current
    I_ext_list = [[Iext] for _ in range(steps)]
    for t in range(steps):
        neuron.update_inputs(Iext, np.array([]), np.array([]))
        neuron.update_state(dt)
    features = extract_features(np.array(neuron.Vvalues), dt)
    return {"Vs0": Vs0, "Vus0": Vus0, "Iext": Iext, "g_us": g_us, **features}

if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(__file__), "param_sweep")
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    # Prepare parameter grid
    grid = list(itertools.product(taus0_range, vus0_range, Iext_range, gus_range))
    # Parallel execution
    with Pool() as pool:
        results = pool.map(run_sim, grid)
    # Save summary to CSV
    df = pd.DataFrame(results)
    df.to_csv(f"{filepath}/sweep_summary.csv", index=False)
    # Save traces of select runs to HDF5
    with h5py.File(f"{filepath}sweep_traces.h5", "w") as hf:
        for idx, params in enumerate(grid):
            Vs0, Vus0, Iext, g_us = params
            grp = hf.create_group(f"run_{idx}")
            grp.attrs.update({"Vs0": Vs0, "Vus0": Vus0, "Iext": Iext, "g_us": g_us})
            grp.create_dataset("V_trace", data=np.array(results[idx]["V_trace"]) if "V_trace" in results[idx] else np.array([]))
    print("Parameter sweep completed. Results written to sweep_summary.csv and sweep_traces.h5")